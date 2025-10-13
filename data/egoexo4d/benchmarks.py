import copy
import itertools
import math
import os
import random
import Levenshtein as lev
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizer
from torchcodec.decoders import VideoDecoder

from ..stream import StreamMixIn
from ..utils import DictWithTo, fixed_sampling

from .egoexo4d import EGOEXO4D


class EGOEXO4DBenchmark(EGOEXO4D, StreamMixIn):
    evaluation_kwargs = DictWithTo(
        evaluator="generate_after_embed",
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        temperature=1.0,
        top_p=1.0,
    )

    @staticmethod
    def fuzzy_match(text, choices):
        scores = [-lev.distance(text, choice) for choice in choices]
        return scores.index(max(scores))

    def compute_metrics(
        self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs
    ):
        out_dir = kwargs["output_dir"]
        batch_pred_tensor, sample_idxs = (
            eval_predictions.predictions,
            eval_predictions.label_ids,
        )
        batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id
        predictions = tokenizer.batch_decode(
            batch_pred_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if ":" in predictions[0]:
            with open(f"{out_dir}/outputs.txt", "w") as f_out:
                f_out.write("\n".join(predictions))
            predictions = [text.split(":")[1].strip() for text in predictions]

        os.makedirs(out_dir, exist_ok=True)
        if self.split == "test":
            with open(f"{out_dir}/test_predictions.txt", "w") as f_pred:
                f_pred.write("\n".join(predictions))
            return dict(accuracy=0.0)

        # [if validation i.e. labels are available] output labels and print the accuracy
        with open(f"{out_dir}/preds.txt", "w") as f_pred:
            f_pred.write("\n".join(predictions))

        with open(f"{out_dir}/labels.txt", "w") as f_labels:
            f_labels.write(
                "\n".join(
                    f"{label}: {self.mapping_categories[label]}"
                    for label in self.answers
                )
            )

        predictions = np.array(
            [self.fuzzy_match(text, self.mapping_categories) for text in predictions]
        )
        accuracy = (predictions == np.array(self.answers)).mean()
        return dict(accuracy=accuracy * 100)

    def __sample_frames(self, start, end, video_uid, fps, length):
        fps_ratio = fps / self.anno_fps
        assert (
            self.num_samples > 0
        ), f"EgoExo4D Keystep benchmark required fixed frame sample. Set num_samples > 0. Currently num_samples = {self.num_samples}"

        frames = fixed_sampling(self.split, self.num_samples, start, end)
        frames = (frames * fps_ratio).astype(int)
        frames = np.clip(frames, 1, length - 1)

        return frames

    def __getitem__(self, index):
        anno = self.annos[index]
        conversation = anno.pop("conversation")
        frames = anno.pop("frames")
        video_path = anno.pop("video_path")

        vpath = os.path.join(video_path, "frame_aligned_videos/downscaled/448")
        videos = [f for f in os.listdir(vpath) if "214-1" in f and f.endswith(".mp4")]
        if len(videos) == 0:
            raise FileNotFoundError(f"No video found in {vpath} for index {index}.")
        else:
            vpath = os.path.join(vpath, videos[0])

        record = VideoDecoder(vpath, device="cpu", dimension_order="NHWC")

        frames["length"] = record.metadata.num_frames
        frames["fps"] = record.metadata.average_fps

        frames = self.__sample_frames(**frames)

        conversation[-2]["num_frames"] = len(frames)
        conversation[-2]["long_context"] = [""]
        conversation = (
            conversation if self.is_training else conversation[:-1]
        )  # if not training, do not include the assistant message

        load_ranges = {video_path: frames}

        return (
            *super().__getitem__(
                conversation=conversation,
                load_ranges=load_ranges,
                record=record,
                add_generation_prompt=not self.is_training,
            ),
            index,
            self.evaluation_kwargs,
        )


class EGOEXO4DKeystep(EGOEXO4DBenchmark):
    random.seed(42)

    def __init__(
        self,
        *,
        split: str,
        frame_fps: int,
        num_samples: int,
        is_training: bool,
        transform: None,
        **kwargs,
    ):
        super().__init__(
            split=split,
            frame_fps=frame_fps,
            num_samples=num_samples,
            is_training=is_training,
            **kwargs,
        )
        self.split = split
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.anno_fps = 30
        self.annos = []
        self.answers, self.mapping_categories = [], self.step_categories
        self.num_samples = num_samples
        self.transform = transform

        user_message = {
            "role": "user",
            "content": "Describe the activity step being performed in the video. Format your answer concisely. No extra text output.",
        }

        if self.split == "trainval":
            self.split = "train"

        for anno in self._annos:
            video_uid = anno["video_uid"]
            step = anno["step"]
            answer = anno["step"]
            response = answer

            start_frame = max(anno["start_frame"], 1)
            end_frame = anno["end_frame"]
            end_frame = start_frame + 1 if start_frame >= end_frame else end_frame

            conversation = [
                user_message,
                {"role": "stream"},
                {"role": "assistant", "content": response},
            ]

            if is_training:
                conversation[-1]["learn"] = True
                conversation[-2]["learn"] = True

            self.annos.append(
                {
                    "conversation": conversation,
                    "frames": {
                        "start": start_frame,
                        "end": end_frame,
                        "video_uid": video_uid,
                    },
                    "video_path": anno["video_path"],
                    "index": anno["index"],
                }
            )

            if not self.split == "test":
                self.answers.append(self.mapping_categories.index(answer))
        print(f"Total {self.split} samples: {len(self.annos)}")


def build_egoexo4d_keystep_train(**kwargs):
    return EGOEXO4DKeystep(split="train", **kwargs)


def build_egoexo4d_keystep_trainval(**kwargs):
    return EGOEXO4DKeystep(split="trainval", **kwargs)


def build_egoexo4d_keystep_val(**kwargs):
    return EGOEXO4DKeystep(split="val", **kwargs)


def build_egoexo4d_keystep_test(**kwargs):
    return EGOEXO4DKeystep(split="test", **kwargs)
