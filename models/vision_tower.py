import math, torch, os
from functools import partial
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from .configuration_live import LiveConfigMixin


def build_model_tokenizer(
    model, path=None, attn_implementation="sdpa", torch_dtype="auto"
):
    print(f"Adding vision encoder: {model}")

    model_id = os.path.join(path, model) if path else model

    if "siglip2" in model:
        from transformers import (
            AutoModelForZeroShotImageClassification,
            AutoProcessor,
            AutoImageProcessor,
        )

        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

        model = AutoModelForZeroShotImageClassification.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        return model.vision_model.eval(), processor
    else:
        raise ValueError(
            f"{model} not supported. Please implement support for {model} in models/vision_tower.py"
        )


def _siglip_vision_encode(
    model,
    frames: Tensor,
    frame_token_cls: bool = None,
    frame_token_pooled: list[int] = None,
    **kwargs,
):
    # splits batch into batches of size 64 to avoid GPU OOM for long videos
    with torch.inference_mode():
        outputs = [
            model(frames[i : min(i + 64, len(frames))])
            for i in range(0, frames.size(0), 64)
        ]
        last_hidden_state = torch.cat(
            [x.last_hidden_state.detach() for x in outputs], dim=0
        )
        pooler_output = torch.cat([x.pooler_output.detach() for x in outputs], dim=0)

    spatial_tokens = None
    if frame_token_pooled != 0:
        if frame_token_pooled == -1:
            spatial_tokens = last_hidden_state.detach().clone()
        elif frame_token_pooled > 0:
            s = int(
                math.sqrt(last_hidden_state.shape[1])
            )  # Calculate size of square grid, excluding CLS token

            if s % frame_token_pooled != 0 or s % frame_token_pooled != 0:
                raise ValueError(
                    "The size of the patches is not divisible by the desired pooling output size."
                )
            spatial_tokens = (
                torch.nn.functional.adaptive_avg_pool2d(
                    last_hidden_state.reshape(
                        last_hidden_state.shape[0],
                        s,
                        s,
                        last_hidden_state.shape[-1],
                    ).permute(0, 3, 1, 2),
                    frame_token_pooled,
                )
                .flatten(2, 3)
                .permute(0, 2, 1)
            )

    combined_features = []
    if frame_token_cls:
        combined_features.append(
            pooler_output.unsqueeze(1)
        )  # Shape: (batch_size, 1, d) --> [CLS] token

    if spatial_tokens is not None:
        combined_features.append(spatial_tokens)

    combined_features = torch.cat(combined_features, dim=1)  # Shape: (batch_size, n, d)
    return combined_features


def _vision_encode(
    model,
    frames: Tensor,
    model_name: str,
    frame_token_cls: bool = None,
    frame_token_pooled: list[int] = None,
    **kwargs,
):
    if "siglip" in model_name:
        return _siglip_vision_encode(model, frames, frame_token_cls, frame_token_pooled)
    else:
        raise ValueError(
            f"{model} not supported. Please implement support for {model} in models/vision_tower.py"
        )


def build_vision_tower(
    config: LiveConfigMixin, attn_implementation="sdpa", torch_dtype="auto"
):
    encoder, processor = build_model_tokenizer(
        config.vision_pretrained, None, attn_implementation, torch_dtype
    )
    return encoder, processor, partial(_vision_encode)
