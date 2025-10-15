import torch, os, re, json, random, math, copy, itertools, psutil
import Levenshtein as lev
import numpy as np
import pickle as pkl
from transformers import EvalPrediction, PreTrainedTokenizer
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchcodec.decoders import VideoDecoder
from torchaudio.utils import ffmpeg_utils

from ..utils import DictWithTo

from ..stream import StreamMixIn

from .egoclip import EgoClip

random.seed(42)
np.random.seed(42)
ffmpeg_utils.set_log_level(-8)


class EgoClipStage1(EgoClip, StreamMixIn):
    evaluation_kwargs = DictWithTo(
        evaluator="generate_after_embed",
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        temperature=1.0,
        top_p=1.0,
    )
    crop_with_boxes = None
    ego_prompts = [
        {"role": "user", "content": "What am I doing in the video?"},
        {"role": "user", "content": "What is the camera wearer doing in the video?"},
        {"role": "user", "content": "What is the person doing in the video?"},
        {"role": "user", "content": "Describe what I am doing in this clip."},
        {
            "role": "user",
            "content": "Can you tell what the camera wearer is doing right now?",
        },
    ]

    nonego_prompts = [
        {"role": "user", "content": "What is happening in the video?"},
        {"role": "user", "content": "What is the person doing in the video?"},
        {"role": "user", "content": "What activity is taking place in the video?"},
        {"role": "user", "content": "Can you describe the action shown in this clip?"},
        {
            "role": "user",
            "content": "Summarize what is occurring in the video right now.",
        },
    ]

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
        self.num_samples = num_samples
        self.transform = transform

        print(f"Total {self.split} samples: {len(self.annos)}")

    def __getitem__(self, index):
        sample = self.annos.iloc[index]
        video_path, video_sec, bound_sec = self.get_video_path(sample)

        response = sample["clip_text"]
        if "#C" in response:
            user_message = random.choice(self.ego_prompts)
        else:
            user_message = random.choice(self.nonego_prompts)

        conversation = [
            user_message,
            {"role": "stream"},
            {"role": "assistant", "content": self.clean_response(response)},
        ]

        conversation[-1]["learn"] = True
        conversation[-2]["learn"] = True

        bboxes, success = self.load_hand_object_bbox(sample)

        frames, crop_params, _, seconds = self.get_video_frames(
            video_path,
            video_sec,
            bound_sec,
            boxes=(bboxes if self.crop_with_boxes else None),
            transform=self.transform,
        )

        conversation[-2]["num_frames"] = 4
        conversation[-2]["long_context"] = [""]
        conversation = (
            conversation if self.is_training else conversation[:-1]
        )  # if not training, do not include the assistant message

        return (
            *super().__getitem__(
                conversation=conversation,
                load_ranges=frames,
                bboxes=bboxes,
                add_generation_prompt=not self.is_training,
            ),
            index,
            self.evaluation_kwargs,
        )

    def clean_response(self, text: str) -> str:
        # 1. Replace known tags (#C, #O) with intended forms or blanks
        text = text.replace("#C C", "The person")
        text = text.replace("#C ", "the person")
        text = text.replace("#O ", "")

        # 2. Collapse multiple spaces created by removals
        text = re.sub(r"\s+", " ", text).strip()

        # 3. Fix lowercase after sentence boundaries (if tag removal broke capitalization)
        # e.g., "The person is cooking. he is stirring." → "The person is cooking. He is stirring."
        text = re.sub(
            r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
        )

        # 4. Ensure first character is capitalized (if start lost capitalization)
        text = text[0].upper() + text[1:] if text else text

        return text

    def get_video_path(self, sample):
        video_uid = sample["video_uid"]
        video_start_sec = max(float(sample["clip_start"]), 0)
        video_end_sec = max(float(sample["clip_end"]), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(
            self.video_root, video_uid, str(chunk_start_id) + ".mp4"
        )
        full_video_end_fp = os.path.join(
            self.video_root, video_uid, str(chunk_end_id) + ".mp4"
        )

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec

        return video_fp, video_sec, bound_sec

    def load_hand_object_bbox(self, sample):
        clip_start = float(sample["clip_start"])

        hand_boxes = torch.zeros(4, 2, 4)
        obj_boxes = torch.zeros(4, 4, 4)
        image_size = (0, 0)

        video_name = sample["video_uid"]
        clip_index = str(int(clip_start // self.chunk_sec))
        hand_file = os.path.join(
            self.handobj_dir, video_name, clip_index + ".handobj.pkl"
        )
        success = 0
        if os.path.exists(hand_file):
            hand_info = pkl.load(open(hand_file, "rb"))

            poss_starts = [clip_start, clip_start - 0.001, clip_start + 0.001]
            image_size = (
                [*hand_info.values()][0]["info"]["height"],
                [*hand_info.values()][0]["info"]["width"],
            )
            for start in poss_starts:
                try:
                    hand_boxes = torch.stack(
                        [load_bboxes(hand_info[round(start, 3)], i) for i in range(4)]
                    )
                    obj_boxes = torch.stack(
                        [
                            load_bboxes(
                                hand_info[round(start, 3)], i, box_type="obj_dets"
                            )
                            for i in range(4)
                        ]
                    )
                    success = 1
                except:
                    success = 0
                if success == 1:
                    break

        boxes = torch.cat([hand_boxes, obj_boxes], 1)
        boxes = boxes / torch.tensor(
            [image_size[1], image_size[0], image_size[1], image_size[0]],
            dtype=boxes.dtype,
            device=boxes.device,
        )

        return boxes, success

    def get_video_frames(
        self, video_fp, video_sec, bound_sec, boxes=None, pred=False, transform=False
    ):
        video_params = {"input_res": 224, "num_frames": 4, "loading": "lax"}
        video_loading = video_params.get("loading", "loose")
        try:
            # assert False
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                images, seconds = read_frames_cv2_egoclip_torchcodec(
                    video_fp[0],
                    video_sec[0],
                    end_second=video_sec[1],
                    clip_length=video_params["num_frames"],
                )
                valid = 1
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == "strict":
                raise ValueError(
                    f"Video loading failed for {video_fp}, video loading for this dataset is strict."
                ) from e
            else:
                img = Image.new(
                    "RGB",
                    (video_params["input_res"], video_params["input_res"]),
                    (0, 0, 0),
                )
                images = [img.copy() for _ in range(video_params["num_frames"])]
                valid = 0
                seconds = [0.0] * video_params["num_frames"]
                print(f"[WARN] torchcodec failed for {video_fp}: {e}")

        crop_params = torch.tensor([0.0, 0.0, 0.0, 0.0])
        if transform:
            if hasattr(transform, "tokenizer"):
                frames = transform(
                    text="A", images=images, return_tensors="pt", padding=True
                )["pixel_values"]
            else:
                frames = transform(images=images, return_tensors="pt")["pixel_values"]
        else:
            raise ValueError("Transform must be provided for video frames.")

        return frames, crop_params, valid, seconds


def build_egoclip_stage1(**kwargs):
    return EgoClipStage1(split="train", **kwargs)


def load_bboxes(hand_info, ind, box_type="hand_dets", threshold=0.5):
    ind = ind % 600
    if box_type == "hand_dets":
        max_boxes = 2
    else:
        max_boxes = 4
    out_boxes = torch.zeros(max_boxes, 4)
    if int(ind) in hand_info:
        dets = hand_info[int(ind)][box_type]
        if dets is not None:
            boxes = torch.tensor(dets[:, :4])
            scores = torch.tensor(dets[:, 4:5])[:, 0]
            boxes, scores = boxes[scores > 0.5], scores[scores > 0.5]
            topk = torch.argsort(scores, descending=True)[:max_boxes]
            out_boxes[: len(topk)] = boxes[topk]
    return out_boxes


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def read_frames_cv2_egoclip_torchcodec(
    vpath,
    start_second,
    end_second=None,
    chunk_len=600,
    fps=30,
    clip_length=32,
    jitter=False,
):
    dec = VideoDecoder(vpath, device="cpu", dimension_order="NHWC")
    frame_count = dec.metadata.num_frames
    stream_fps = dec.metadata.average_fps

    eff_fps = float(stream_fps) if (fps == -1 and stream_fps) else float(fps)
    if eff_fps <= 0:
        eff_fps = 30.0

    duration_guess = None
    if frame_count is not None and frame_count > 0:
        duration_guess = frame_count / eff_fps

    if chunk_len == -1:
        # whole video; clamp end to duration if known
        if end_second is None:
            end_second = (
                len(dec) / eff_fps
                if frame_count is not None
                else (start_second + clip_length / eff_fps)
            )
        if duration_guess is not None:
            end_second = min(end_second, duration_guess)
        second_offset = float(start_second)  # local == global
    else:
        chunk_start = (int(start_second) // int(chunk_len)) * int(chunk_len)
        second_offset = float(start_second) - float(chunk_start)
        if end_second is None:
            end_second = start_second + (clip_length / eff_fps)

    frame_offset = int(np.round(second_offset * eff_fps))
    total_duration = max(int((end_second - start_second) * eff_fps), clip_length)

    if chunk_len == -1:
        if end_second <= start_second:
            raise ValueError("end_second should be greater than start_second")
        start_idx = frame_offset
        end_idx_excl = min(
            frame_offset + total_duration,
            len(dec) if frame_count is not None else (frame_offset + total_duration),
        )
    else:
        start_idx = frame_offset
        end_idx_excl = frame_offset + total_duration

    frame_ids = get_frame_ids(
        start_idx,
        end_idx_excl,
        num_segments=clip_length,
        jitter=jitter,
    )

    # --- decode exact indices; handle tail overflow by duplicating last valid frame ---
    n = len(dec) if frame_count is not None else None
    # split indices into valid and overflow
    if n is not None:
        valid_ids = [i for i in frame_ids if 0 <= i < n]
        overflow_count = len(frame_ids) - len(valid_ids)
    else:
        valid_ids = frame_ids
        overflow_count = 0

    try:
        frames_t = dec.get_frames_at(valid_ids).data
    except Exception as e:
        chunks = []
        for i in valid_ids:
            try:
                chunks.append(dec.get_frames_at([i]).data)  # shape [1,H,W,C]
            except Exception:
                pass
        if not chunks:
            raise
        frames_t = torch.cat(chunks, dim=0)  # [V,H,W,C]

    t_got = frames_t.shape[0]
    if t_got < clip_length:
        if t_got == 0:
            H = W = 224
            frames_t = torch.zeros((1, H, W, 3), dtype=torch.uint8)
            t_got = 1
        pad = frames_t[-1:].repeat(clip_length - t_got, 1, 1, 1)
        frames_t = torch.cat([frames_t, pad], dim=0)

    frames = frames_t.cpu().numpy()  # [T,H,W,C], uint8
    times = [f / 30.0 for f in frame_ids]

    return frames, times
