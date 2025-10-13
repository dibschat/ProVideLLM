import copy
import itertools
import math
import os
import random
import Levenshtein
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizer
from torchcodec.decoders import VideoDecoder

from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, DictWithTo, fixed_sampling

from .coin import COIN


class COINBenchmark(COIN, StreamMixIn):
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
        return min(
            [(Levenshtein.distance(text, choice), choice) for choice in choices]
        )[1]

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

        os.makedirs(out_dir, exist_ok=True)

        correct = 0
        with open(f"{out_dir}/predictions.txt", "w") as f:
            for prediction, label in zip(predictions, self.labels[sample_idxs]):
                prediction = prediction.lower().strip()
                if (
                    prediction == label
                    or self.fuzzy_match(prediction, self.categories) == label
                ):
                    correct += 1
                f.write(f"{prediction} | {label}\n")

        return dict(accuracy=correct / len(predictions) * 100)  # * 100

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

        record = VideoDecoder(video_path, device="cpu", dimension_order="NHWC")

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


class COINStep(COINBenchmark):
    random.seed(42)
    user_message = {
        "role": "user",
        "content": "What is the action in the video? Format your answer concisely. No extra text output.",
    }

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
        self.annos, self.labels = [], []
        self.num_samples = num_samples
        self.transform = transform

        for anno in self._annos:
            video_uid = anno["video_uid"]

            steps = anno["steps"]
            for i in range(len(steps)):
                response = steps[i]["text"]
                self.labels.append(steps[i]["text"].lower())

                start_frame = max(steps[i]["start_frame"], 1)
                end_frame = steps[i]["end_frame"]
                end_frame = start_frame + 1 if start_frame >= end_frame else end_frame

                conversation = [
                    COINStep.user_message,
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
                    }
                )
        self.labels = np.array(self.labels)
        self.categories = self.step_categories

        print(f"Total {self.split} samples: {len(self.annos)}")


def build_coin_step_train(**kwargs):
    return COINStep(split="train", **kwargs)


def build_coin_step_test(**kwargs):
    return COINStep(split="test", **kwargs)


class COINNext(COINBenchmark):
    random.seed(42)
    user_message = {
        "role": "user",
        "content": "What is the next action for the video? Format your answer concisely. No extra text output.",
    }

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
        self.annos, self.labels = [], []
        self.num_samples = num_samples
        self.transform = transform

        for anno in self._annos:
            video_uid = anno["video_uid"]
            steps = anno["steps"]
            start_frame = max(anno["start_frame"], 1)

            for i in range(len(steps) - 1):
                response = steps[i + 1]["text"]
                self.labels.append(steps[i + 1]["text"].lower())

                end_frame = steps[i]["end_frame"]
                end_frame = start_frame + 1 if start_frame >= end_frame else end_frame

                conversation = [
                    COINNext.user_message,
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
                    }
                )
        self.labels = np.array(self.labels)
        self.categories = self.step_categories

        print(f"Total {self.split} samples: {len(self.annos)}")


def build_coin_next_train(**kwargs):
    return COINNext(split="train", **kwargs)


def build_coin_next_test(**kwargs):
    return COINNext(split="test", **kwargs)


class COINTask(COINBenchmark):
    user_message = {
        "role": "user",
        "content": "What is the overall activity in the video? Format your answer concisely. No extra text output.",
    }

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
        self.annos, self.labels = [], []
        self.num_samples = num_samples
        self.transform = transform

        for anno in self._annos:
            video_uid = anno["video_uid"]
            response = anno["task"]
            self.labels.append(anno["task"].lower())

            start_frame = max(anno["start_frame"], 1)
            end_frame = anno["end_frame"]
            end_frame = start_frame + 1 if start_frame >= end_frame else end_frame

            conversation = [
                COINTask.user_message,
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
                }
            )
        self.labels = np.array(self.labels)
        self.categories = self.task_categories

        print(f"Total {self.split} samples: {len(self.annos)}")


def build_coin_task_train(**kwargs):
    return COINTask(split="train", **kwargs)


def build_coin_task_test(**kwargs):
    return COINTask(split="test", **kwargs)


class COINProcedure(COINBenchmark):
    random.seed(42)
    max_num_steps = 5
    user_message = lambda num_steps: {
        "role": "user",
        "content": f"What are the next {num_steps} actions for the video? Format your answer concisely, listing each action separated by a ';'. No extra text output.",
    }

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
        self.annos, self.labels = [], []
        self.num_samples = num_samples
        self.transform = transform

        for anno in self._annos:
            video_uid = anno["video_uid"]
            steps = anno["steps"]
            start_frame = max(anno["start_frame"], 1)

            for i in range(len(steps) - 1):
                end_frame = steps[i]["end_frame"]
                end_frame = start_frame + 1 if start_frame >= end_frame else end_frame

                next_steps = steps[i + 1 : i + self.max_num_steps + 1]
                num_next_steps = len(next_steps)

                if num_next_steps == 1:
                    response = next_steps[0]["text"].lower()
                    self.labels.append(next_steps[0]["text"].lower())
                    conversation = [
                        COINNext.user_message,
                        {"role": "stream"},
                        {"role": "assistant", "content": response},
                    ]
                else:
                    response = "; ".join(f"{s['text'].lower()}" for s in next_steps)
                    self.labels.append(response)
                    conversation = [
                        COINProcedure.user_message(num_next_steps),
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
                    }
                )
        self.categories = self.step_categories

        print(f"Total {self.split} samples: {len(self.annos)}")

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

        os.makedirs(out_dir, exist_ok=True)
        correct, total = 0, 0
        labels = [self.labels[i] for i in sample_idxs]
        with open(f"{out_dir}/predictions_procedure.txt", "w") as f:
            for prediction_steps, label_steps in zip(predictions, labels):
                line = f"{prediction_steps} | {label_steps}"
                f.write(line + "\n")

                for prediction_step, label_step in zip(
                    prediction_steps.split("; "), label_steps.split("; ")
                ):
                    prediction_step = prediction_step.strip()
                    if (
                        prediction_step == label_step
                        or self.fuzzy_match(prediction_step, self.categories)
                        == label_step
                    ):
                        correct += 1
                    total += 1
        return {"accuracy": correct / total * 100}


def build_coin_procedure_train(**kwargs):
    return COINProcedure(split="train", **kwargs)


def build_coin_procedure_test(**kwargs):
    return COINProcedure(split="test", **kwargs)


class COINTaskProcedure(COINBenchmark):
    random.seed(42)
    max_num_steps = 5
    get_query_single = lambda task: {
        "role": "user",
        "content": f"To {task}, what is the next action for the video? Format your answer concisely. No extra text output.",
    }
    get_query_multi = lambda task, num_steps: {
        "role": "user",
        "content": f"To {task}, what is the next {num_steps} actions for the video? Format your answer concisely, listing each action separated by a ';'. No extra text output.",
    }

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
        self.annos, self.labels = [], []
        self.num_samples = num_samples
        self.transform = transform

        for anno in self._annos:
            video_uid = anno["video_uid"]
            steps = anno["steps"]
            start_frame = max(anno["start_frame"], 1)

            for i in range(len(steps) - 1):
                end_frame = steps[i]["end_frame"]
                end_frame = start_frame + 1 if start_frame >= end_frame else end_frame

                next_steps = steps[i + 1 : i + self.max_num_steps + 1]
                num_next_steps = len(next_steps)

                if num_next_steps == 1:
                    response = next_steps[0]["text"].lower()
                    self.labels.append(next_steps[0]["text"].lower())
                    conversation = [
                        COINTaskProcedure.get_query_single(anno["task"].lower()),
                        {"role": "stream"},
                        {"role": "assistant", "content": response},
                    ]
                else:
                    response = "; ".join(f"{s['text'].lower()}" for s in next_steps)
                    self.labels.append(response)

                    conversation = [
                        COINTaskProcedure.get_query_multi(
                            anno["task"].lower(), num_next_steps
                        ),
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
                    }
                )
        self.categories = self.step_categories

        print(f"Total {self.split} samples: {len(self.annos)}")

    def compute_metrics(self, *args, **kwargs):
        return COINProcedure.compute_metrics(self, *args, **kwargs)


def build_coin_taskprocedure_train(**kwargs):
    return COINTaskProcedure(split="train", **kwargs)


def build_coin_taskprocedure_test(**kwargs):
    return COINTaskProcedure(split="test", **kwargs)
