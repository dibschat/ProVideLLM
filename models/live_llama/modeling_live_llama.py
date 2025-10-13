import torch
from torch import nn
from transformers import Cache, LlamaForCausalLM
from transformers.activations import GELUActivation
from transformers.utils import logging

from ..modeling_live import build_live, LiveMixin

from .configuration_live_llama import LiveLlamaConfig

from .connector import MLP, DetrQFormer
from .box_utils import build_matcher, SetCriterion, compute_box_loss

logger = logging.get_logger(__name__)


class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ["vision_encoder", "connector"]

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        self.model_variant = config.model_variant
        self.stage = config.stage
        self.N_s = config.N_s
        self.N_l = config.N_l
        self.background_token_id = 4092
        if config.connector_type == "detr_qformer":
            self.connector = nn.ModuleList()
            self.connector.append(
                MLP(
                    input_dim=config.vision_hidden_size,
                    hidden_dim=config.hidden_size,
                    output_dim=config.hidden_size,
                    num_layers=2,
                    activation="gelu",
                )
            )
            self.connector.append(
                DetrQFormer(
                    stage=config.stage,
                    is_training=getattr(config, "is_training", False),
                    num_queries=config.compressed_tokens,
                    hand_queries=config.hand_tokens,
                    obj_queries=config.object_tokens,
                    input_dim=config.vision_hidden_size,
                    output_dim=config.hidden_size,
                    num_patch_tokens=config.vision_num_tokens,
                    nhead=config.connector_nhead,
                    num_layers=config.connector_num_layers,
                    hidden_dim=config.connector_hidden_dim,
                    activation="gelu",
                    aux_loss=True,
                )
            )
        elif config.connector_type == "MLP":
            self.connector = MLP(
                input_dim=config.vision_hidden_size,
                hidden_dim=config.hidden_size,
                output_dim=config.hidden_size,
                num_layers=2,
                activation="gelu",
            )
        else:
            raise ValueError(
                f"[{config.connector_type}] connector is not supported for."
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        bboxes: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds, det, meta = self.joint_embed(input_ids, frames)

        outputs = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        lm_loss, bbox_loss = None, None

        # ----- LM loss -----
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id

            weight = v_mask * self.config.stream_loss_weight + ~v_mask

            lm_loss = (
                nn.functional.cross_entropy(
                    logits.flatten(0, 1), labels.flatten(), reduction="none"
                )
                * weight
            )
            lm_loss = lm_loss.sum() / ((labels.flatten()[weight.bool()]) >= 0).sum()

        # ----- BBox loss -----
        bbox_loss = None
        box_w = 0.0
        if bboxes is not None:
            bbox_loss = self.compute_bbox_loss(bboxes, det, meta)
            box_w = getattr(self.config, "box_loss_weight", 1.0)

        loss = (
            ((lm_loss or 0.0) + box_w * (bbox_loss or 0.0))
            if (lm_loss is not None or bbox_loss is not None)
            else None
        )

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs

        outputs.loss = loss
        return outputs

    def generate_after_embed(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        **kwargs,
    ):
        outputs = self.joint_embed(input_ids, frames)
        return (
            super().generate(inputs_embeds=outputs[0], **kwargs),
            outputs[1],
            outputs[2],
        )

    def compute_bbox_loss(self, bboxes, det, meta):
        """
        Compute total bbox loss = hand + object
        - bboxes: [B, 6, 4] normalized XYXY (0..1)
        - det: {'pred_boxes': [B, 6, 4] cxcywh (0..1), 'aux_outputs': [{'pred_boxes': ...}, ...]}
        - meta: unused here (kept for symmetry/future use)
        Returns: scalar loss (tensor)
        """

        if bboxes is None or det is None or det.get("pred_boxes") is None:
            return det["pred_boxes"].sum() * 0.0

        device = det["pred_boxes"].device
        dtype = det["pred_boxes"].dtype
        B, Q, _ = det["pred_boxes"].shape

        detr_out = {
            "pred_boxes": det["pred_boxes"],
            "aux_outputs": [
                {"pred_boxes": a["pred_boxes"]} for a in det.get("aux_outputs", [])
            ],
        }

        H, W = getattr(self, "box_grid", (224, 224))
        scale = torch.tensor([W, H, W, H], device=device, dtype=dtype)

        hand_boxes_px = bboxes[:, :2, :] * scale
        obj_boxes_px = bboxes[:, 2:, :] * scale

        all_image_size = torch.tensor([H, W], device=device, dtype=dtype).expand(B, 2)

        if not hasattr(self, "_hh_matcher"):
            self._hh_matcher = build_matcher(None)

        if not hasattr(self, "_hh_criterion"):
            weight_dict = {
                "loss_bbox_hand_boxes": 5,
                "loss_giou_hand_boxes": 2,
                "loss_bbox_obj_boxes": 5,
                "loss_giou_obj_boxes": 2,
            }
            self._hh_criterion = SetCriterion(
                num_classes=1,
                matcher=self._hh_matcher,
                weight_dict=weight_dict,
                eos_coef=0.1,
                losses=["boxes"],
            )

        n_q = Q

        loss_hand, _ = compute_box_loss(
            "hand_boxes",
            self._hh_criterion,
            detr_out,
            target_boxes=hand_boxes_px,
            target_classes=None,
            all_image_size=all_image_size,
            n_queries=n_q,
        )
        loss_obj, _ = compute_box_loss(
            "obj_boxes",
            self._hh_criterion,
            detr_out,
            target_boxes=obj_boxes_px,
            target_classes=None,
            all_image_size=all_image_size,
            n_queries=n_q,
        )

        return loss_hand + loss_obj


def build_live_llama(**kwargs):
    return build_live(
        config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs
    )


if __name__ == "__main__":
    from ..arguments_live import LiveOnePlusTrainingArguments

    print(LiveOnePlusTrainingArguments().to_dict())
    model, tokenizer = build_live_llama(
        is_training=True, **LiveOnePlusTrainingArguments().to_dict()
    )
    print(model.config, tokenizer)
