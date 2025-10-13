from functools import partial

import torch
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother


def data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    (
        batch_text,
        batch_frames,
        batch_learn_ranges,
        batch_bboxes,
        batch_load_ranges,
        batch_sample_idx,
        batch_evaluation_kwargs,
    ) = batch

    if len(batch_text) == 1 and isinstance(batch_text[0], list):
        batch = {"tokenizer": tokenizer}
    else:
        batch = tokenizer(
            batch_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )

        batch_labels = torch.full_like(
            batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long
        )

        for text, labels, input_ids, offset_mapping, learn_range in zip(
            batch_text,
            batch_labels,
            batch.input_ids,
            batch.offset_mapping,
            batch_learn_ranges,
        ):
            output = []
            for learn_r in learn_range:
                # Find the minimum index in offset_mapping[:, 0] that is >= learn_r.start
                start_candidates = torch.nonzero(
                    offset_mapping[:, 0] >= learn_r.start, as_tuple=False
                )
                start = (
                    start_candidates.min().item() if start_candidates.numel() > 0 else 0
                )

                # Find the minimum closest stop index
                if offset_mapping[:, 0][-1] >= learn_r.stop:
                    stop_candidates = torch.nonzero(
                        offset_mapping[:, 1] <= learn_r.stop, as_tuple=False
                    )
                    stop = (
                        stop_candidates.max().item()
                        if stop_candidates.numel() > 0
                        else len(input_ids)
                    )
                    if start == stop:
                        stop = stop + 1 if stop < len(input_ids) else start - 1
                else:  # the last eos token
                    stop = len(input_ids)
                labels[start - 1 : stop - 1] = input_ids[start:stop]
                # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
                # this is because some frames has v_placeholder_id target. so replace it with eos token.
                labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id

                output.append([start, stop, input_ids[start:stop]])
                if stop == len(input_ids):
                    output[-1].append(labels)

        batch["labels"] = batch_labels
        batch.pop("offset_mapping")

    batch["frames"] = torch.cat(batch_frames)
    if batch_bboxes[0] is not None:
        batch["bboxes"] = torch.cat(batch_bboxes, dim=0)
    else:
        batch["bboxes"] = None

    batch["sample_idxs"] = torch.tensor(batch_sample_idx)
    if batch_evaluation_kwargs[0]:
        if "stream" in batch_evaluation_kwargs[0]["evaluator"]:
            batch_evaluation_kwargs[0]["load_ranges"] = batch_load_ranges[0]
        batch["evaluation_kwargs"] = batch_evaluation_kwargs[
            0
        ]  # evaluation only supports bs = 1, so its okay
    return batch


def get_data_collator(**kwargs):
    return partial(data_collator, **kwargs)
