import os, torch

from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, Cache
from transformers.utils import logging
from tqdm import tqdm
import numpy as np
from PIL import Image

from .tokenization_live import build_live_tokenizer_and_update_config
from .vision_tower import build_vision_tower
from functools import reduce
import operator

logger = logging.get_logger(__name__)


class LiveMixin(AutoModelForCausalLM):
    def attach_vision_tower(self, attn_implementation, torch_dtype):
        logger.warning_once(
            "!!! Original vision encoder is being used since no features are loaded."
        )
        self.vision_encoder, self.vision_processor, self.vision_encode = (
            build_vision_tower(self.config, attn_implementation, torch_dtype)
        )

    def detach_vision_tower(self):
        del self.vision_encoder
        del self.vision_processor
        del self.vision_encode

    def run_connector(self, frames):
        tokens_list, det_list, meta_list = [], [], []

        for connec, v in zip(self.connector, frames):
            out = connec(v)
            if isinstance(out, tuple):
                tok, det, meta = out
                tokens_list.append(tok)
                if det is not None:
                    det_list.append(det)
                if meta is not None:
                    meta_list.append(meta)
            else:
                tokens_list.append(out)

        if self.model_variant == "providellm_1b":
            tokens_list[-1] = torch.cat(
                [
                    tokens_list[-1][:, meta_list[-1]["idx_slices"]["visual"], :],
                    tokens_list[-1][:, meta_list[-1]["idx_slices"]["hand"], :].mean(
                        1, keepdim=True
                    ),
                    tokens_list[-1][:, meta_list[-1]["idx_slices"]["object"], :].mean(
                        1, keepdim=True
                    ),
                ],
                dim=1,
            )

        tokens = torch.cat(tokens_list, dim=-2)

        det = None
        if len(det_list) == 1:
            det = det_list[0]
        elif len(det_list) > 1:
            det = det_list

        meta = None
        if len(meta_list) == 1:
            meta = meta_list[0]
        elif len(meta_list) > 1:
            meta = meta_list

        return tokens, det, meta

    def visual_embed(self, frames: torch.Tensor):

        self.vision_encoder = self.vision_encoder.to(self.device)
        frames = frames.to(self.dtype)

        frames = self.vision_encode(
            self.vision_encoder,
            frames,
            self.config.vision_pretrained,
            1,
            -1,
        )
        frames = [
            frames[:, 0, :].unsqueeze(1).to(self.dtype),
            frames[:, 1:, :].to(self.dtype),
        ]

        det, meta = None, None
        frames, det, meta = self.run_connector(frames)
        return frames.view(-1, frames.shape[-1]), det, meta

    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids), None, None
        if input_ids is None:
            return self.visual_embed(frames), None, None
        inputs_embeds = self.get_input_embeddings()(
            input_ids.clamp(max=self.vocab_size - 1)
        )
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            frames, det, meta = self.visual_embed(frames)
            inputs_embeds[v_mask] = frames
        return inputs_embeds, det, meta


def build_live(
    *,
    is_training: bool,
    config_class: type,
    model_class: type,
    llm_pretrained: str = None,
    fine_tune: str = "lora",
    finetune_modules: list[str] = None,
    lora_modules: str = None,
    lora_r: int = None,
    lora_alpha: int = None,
    resume_from_checkpoint: str = "",
    attn_implementation: str = "flash_attention_2",
    torch_dtype: str | torch.dtype = "auto",
    **kwargs,
):
    pretrained_ckpt_path = kwargs["pretrained_ckpt_path"]
    if pretrained_ckpt_path:
        print(f"Fine-tuning from pre-trained checkpoint: {pretrained_ckpt_path}")

    model_id = llm_pretrained if pretrained_ckpt_path is None else pretrained_ckpt_path

    config = config_class.from_pretrained(model_id, **kwargs)
    compute_dtype = (
        torch.float16
        if kwargs["fp16"]
        else (torch.bfloat16 if kwargs["bf16"] else torch.float32)
    )
    config.is_training = is_training

    model = model_class.from_pretrained(
        model_id,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation,
    )
    tokenizer = build_live_tokenizer_and_update_config(model_id, model.config)

    if is_training:
        if fine_tune == "lora":
            if not (model.base_model.__class__.__name__ == "LoraModel"):
                finetune_modules = [
                    "connector." + name for name, _ in model.connector.named_children()
                ]
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=find_all_linear_names(model),
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                    modules_to_save=finetune_modules,
                    inference_mode=False,
                )
                model = get_peft_model(model, lora_config)
            print(f"Fine-tuning with LoRA...")
        elif fine_tune == "connector":
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)
            print(f"Fine-tuning the connector only...")
        else:
            print("[WARNING!!!] Training the full VideoLLM end-to-end.")
    else:
        if resume_from_checkpoint:
            if fine_tune == "lora":
                model = PeftModel.from_pretrained(
                    model, resume_from_checkpoint, is_trainable=is_training
                )
                print(f"PEFT checkpoint loaded successfully...")
            else:
                model = load_LiveLlamaForCausalLM(resume_from_checkpoint)
                print(f"Checkpoint loaded successfully...")
        else:
            raise ValueError(
                f"!!! Fail to load checkpoint!!! From: [{resume_from_checkpoint}]"
            )
        model = model.to(compute_dtype)
        model.requires_grad_(False)
        model.eval()

    if not hasattr(model, "vision_encoder"):
        model.attach_vision_tower(
            attn_implementation,
            torch_dtype=compute_dtype,
        )
        model.vision_encoder.requires_grad_(False)

    return model, tokenizer


def load_LiveLlamaForCausalLM(resume_from_checkpoint):
    from .live_llama.modeling_live_llama import LiveLlamaForCausalLM

    model = LiveLlamaForCausalLM.from_pretrained(resume_from_checkpoint)
    return model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["connector", "vision_tower"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            if not "lm_head" in name:
                lora_module_names.add(name)

    return list(lora_module_names)
