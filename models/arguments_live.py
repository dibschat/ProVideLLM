from dataclasses import dataclass, field, fields
from transformers import TrainingArguments
from functools import reduce
import operator
import os
import glob


@dataclass
class ProVideLLMBaseTrainingArguments(TrainingArguments):
    system_prompt: str = (
        "You are a multimodal AI assistant that helps users with their daily activities."
        " Below is your conversation with the user, interleaved with the list of video frames provided by the user."
    )

    model_variant: str = "providellm_1b"  # [providellm_1b, providellm_8b]
    stage: int = 2  # [1: pretrain, 2: fine-tune]

    # init dataset
    train_datasets: list[str] = None
    eval_datasets: list[str] = None

    fine_tune: str = "lora"
    lora_modules: str = (
        "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    )
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ["connector"])

    num_samples: int = 0  # [None, fixed_sampling (8/16...)]
    frame_fps: int = 1  # this is ignored if num_samples>0
    frame_token_cls: bool = False
    frame_token_pooled: int = 0

    hand_tokens: int = 2
    object_tokens: int = 2
    box_loss_weight: float = 1.0
    connector_nhead: int = 8
    connector_num_layers: int = 6
    connector_hidden_dim: int = 512

    frame_resolution: int = 384
    frame_token_interval: str = ""
    frame_token_interval_threshold: float = 0.0
    frame_num_tokens: int = 1
    augmentation: bool = False
    max_num_frames: int = 0

    N_s: int = 0
    N_l: int = 0
    stream_loss_weight: float = 0.0

    # init others
    attn_implementation: str = "sdpa"  # [sdpa, flash_attention_2]
    output_dir: str = "outputs/debug"
    dataset_dir: str = "datasets/"
    pretrained_ckpt_path: str = None
    override_output_dir: str = None

    def __post_init__(self):
        super().__post_init__()
        self.set_conditional_variables()
        print(f"CHECKPOINT WILL BE STORED AT: {self.output_dir}")

    def set_conditional_variables(self):
        if self.fine_tune == "lora":
            self.lora_alpha = self.lora_r * 2

        if self.frame_num_tokens > 1:
            self.frame_token_interval = ","
        if self.max_num_frames <= 0:
            self.max_num_frames = 7200 // (
                self.frame_num_tokens + len(self.frame_token_interval)
            )

        if self.override_output_dir:
            self.output_dir = self.override_output_dir
        else:
            dataset = self.train_datasets[0].split("_")[0]
            task = self.train_datasets[0].split("_")[1]
            outDir = f"outputs/{dataset}"
            ckpt_name = f"{task}_{self.model_variant}_stage_{self.stage}"

            ckpt_name += f"_{self.fine_tune}"
            if self.fine_tune == "lora":
                ckpt_name += f"_{self.lora_r}"
            ckpt_name += f"_num_tokens_{self.frame_num_tokens}"
            if self.hand_tokens > 0 or self.object_tokens > 0:
                ckpt_name += f"_hand_{self.hand_tokens}_obj_{self.object_tokens}"
            if self.stage == 1:
                ckpt_name += f"_wt_{self.box_loss_weight}"
            else:
                self.box_loss_weight = 0.0

            if self.num_samples <= 0:
                ckpt_name += f"_fps_{self.frame_fps}"
                if self.N_s > 0:
                    ckpt_name += f"_Ns_{self.N_s}"
                if self.N_l > 0:
                    ckpt_name += f"_Nl_{self.N_l}"
            else:
                ckpt_name += f"_num_samples_{self.num_samples}"
            ckpt_name += f"_epochs_{int(self.num_train_epochs)}"

            # check if checkpoint exists
            exists = glob.glob(os.path.join(outDir, f"{ckpt_name}_run*"))
            if exists:
                ckpt_name = f"{ckpt_name}_run{max([int(f.split('_run')[-1]) for f in exists]) + 1}"
            else:
                ckpt_name = f"{ckpt_name}_run1"

            self.output_dir = os.path.join(outDir, ckpt_name)


@dataclass
class ProVideLLM1BTrainingArguments(ProVideLLMBaseTrainingArguments):
    model_variant: str = "providellm_1b"
    # llm
    llm_pretrained: str = "meta-llama/Llama-3.2-1B-Instruct"
    long_placeholder: str = "<long"

    # vision tower
    vision_pretrained: str = "google/siglip2-base-patch16-384"
    vision_hidden_size: int = 768
    vision_num_tokens: int = 576
    frame_token_cls: bool = True
    frame_token_pooled: int = -1

    # DETR Q-Former
    connector_type: str = "detr_qformer"
    compressed_tokens: int = 2
    hand_tokens: int = 2
    object_tokens: int = 4
    learnable_tokens: int = compressed_tokens + 1 + 1
    box_loss_weight: float = 1.0
    connector_nhead: int = 8
    connector_num_layers: int = 4
    connector_hidden_dim: int = 128

    frame_num_tokens: int = int(frame_token_cls) + learnable_tokens


@dataclass
class ProVideLLM8BTrainingArguments(ProVideLLMBaseTrainingArguments):
    model_variant: str = "providellm_8b"
    # llm
    llm_pretrained: str = "meta-llama/Llama-3.1-8B-Instruct"
    long_placeholder: str = "<long"

    # vision tower
    vision_pretrained: str = "google/siglip2-so400m-patch14-384"
    vision_hidden_size: int = 1152
    vision_num_tokens: int = 729
    frame_token_cls: bool = True
    frame_token_pooled: int = -1

    # DETR Q-Former
    connector_type: str = "detr_qformer"
    compressed_tokens: int = 4
    hand_tokens: int = 2
    object_tokens: int = 4
    learnable_tokens: int = compressed_tokens + hand_tokens + object_tokens
    box_loss_weight: float = 1.0
    connector_nhead: int = 8
    connector_num_layers: int = 6
    connector_hidden_dim: int = 512

    frame_num_tokens: int = int(frame_token_cls) + learnable_tokens


def get_args_class(args):
    if args.model_variant == "providellm_1b":
        return ProVideLLM1BTrainingArguments
    elif args.model_variant == "providellm_8b":
        return ProVideLLM8BTrainingArguments
    else:
        raise NotImplementedError(f"[UNKNOWN MODEL!!!] {args.model_variant}")
