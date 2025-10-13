import glob, json, math, os, re, torch, tqdm
import pandas as pd


class EgoClip:
    root = "datasets/ego4d"
    video_root = os.path.join(root, "videos")
    anno_root = os.path.join(root, "annotations")

    # please put videos that could not be downloaded, or could not be opened with torchcodec
    ignore = {
        "c9c9d2d2-f9cb-405b-b5ab-f48ecf988aaa.mp4",
        "f130564b-a153-4a84-96bf-447cb837272c.mp4",
        "ad71a786-60a3-4be1-b2fb-fc714e061115.mp4",
        "e94b366a-a6dc-4433-a409-c36567ac6ba9.mp4",
        "63a85af7-e27d-438e-90c8-f416efcfb36c.mp4",
        "ecd0d190-ea38-4731-af01-96e05a27ab79.mp4",
        "fb45b5e3-498c-4177-a7f4-161063093014.mp4",
        "0f14d5fb-d911-48aa-9c0d-6bcd10427742.mp4",
        "0e7ba211-0dba-40b8-8ace-a3e5932db4fb.mp4",
        "cb5f5863-555f-495c-badc-f3c29828c1b0.mp4",
        "158629ec-b436-4edb-bcdf-60b4fed8674d.mp4",
        "19dd53a5-1a7f-4342-9849-f251006058af.mp4",
        "6a75b089-b74e-4e45-a345-422587f04f01.mp4",
    }

    def __init__(
        self,
        split: str,
        vision_pretrained: list,
        dataset_dir: str,
        frame_fps: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = os.path.join(dataset_dir, "ego4d")

        self.video_root = os.path.join(self.root, "egoclip/video_chunks")
        self.anno_root = os.path.join(self.root, "egoclip/helping_hands")

        self.frame_fps = frame_fps

        assert split == "train", "EgoClip only supports Stage-1 pretraining."
        self.split = split

        self.chunk_sec = 600  # each video chunk is 10min long
        self.annos, self.annos_by_segment_id, self.handobj_dir = self.get_metadata()

    def __len__(self):
        return len(self.annos)

    def get_metadata(self):
        split_files = {
            "train": "egoclip.csv",
            "val": "egomcq.json",
            "test": "egomcq.json",
        }
        file = split_files[self.split]

        meta_dir = os.path.join(self.anno_root, "metadata/EgoClip")
        handobj_dir = os.path.join(
            self.anno_root, "hand_object_clip_per_video_4f_lavila_narrator_640"
        )

        annos = pd.read_csv(
            os.path.join(meta_dir, file),
            sep="\t",
            on_bad_lines="skip",
        )

        # ignore bad videos
        annos = annos[~annos["video_uid"].astype(str).add(".mp4").isin(self.ignore)]

        annos["segment_id"] = (
            annos["video_uid"]
            + "_"
            + (annos["narration_time"] // self.chunk_sec).astype(str)
        )
        annos_by_segment_id = dict(tuple(annos.groupby("segment_id")))
        print("!!! EgoClip metadata loaded...")

        return annos, annos_by_segment_id, handobj_dir
