import glob, json, math, os, re, torch, tqdm
from functools import reduce


class COIN:
    root = "datasets/coin/"
    video_root = os.path.join(root, "videos")
    anno_root = os.path.join(root, "annotations")

    def __init__(
        self,
        split: str,
        vision_pretrained: list,
        dataset_dir: str,
        frame_fps: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = os.path.join(dataset_dir, "coin")

        self.video_root = os.path.join(self.root, "videos")
        self.anno_root = os.path.join(self.root, "annotations")

        self.frame_fps = frame_fps
        self.anno_fps = 30

        assert split in ["train", "test"]
        self.split = split

        annos = json.load(open(os.path.join(self.anno_root, f"COIN.json")))["database"]
        file_map = self._create_file_map(self.video_root)

        self._annos = [
            {
                "video_uid": video_uid,
                "task": COIN._clean_task(anno["class"]),
                "start_frame": int(anno["start"] * self.anno_fps),
                "end_frame": int(anno["end"] * self.anno_fps),
                "steps": [
                    dict(
                        start_frame=int(step["segment"][0] * self.anno_fps),
                        end_frame=int(step["segment"][1] * self.anno_fps),
                        text=COIN._clean_step(step["label"]),
                    )
                    for step in anno["annotation"]
                ],
                "video_path": file_map[video_uid],
            }
            for video_uid, anno in annos.items()
            if (split in anno["subset"].lower()) and (video_uid in file_map)
        ]

        self.task_categories = list(set([v["task"].lower() for v in self._annos]))
        self.step_categories = list(
            set(
                [
                    step["text"].lower()
                    for steps in self._annos
                    for step in steps["steps"]
                ]
            )
        )

    def get_video_lengths(
        self,
    ):
        metadata_path = f"{self.anno_root}/video_lengths.json"
        if os.path.exists(metadata_path):
            print(f"load {metadata_path}...")
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(
                os.listdir(self.embed_dirs[0]), desc=f"prepare {metadata_path}..."
            ):
                path = os.path.join(self.embed_dirs[0], file)
                length = len(os.listdir(path))
                key = os.path.basename(path).replace("frames_", "")
                metadata[key] = length
            json.dump(metadata, open(metadata_path, "w"), indent=4)
        return metadata

    def __len__(self):
        return len(self.annos)

    @staticmethod
    def _clean_step(step):
        replaces = {
            "process (crop, fold) paper": "crop and fold paper",
            "try to press gun head, spray residual old grease": "try to press gun head to spray residual old grease",
        }
        return replaces.get(step, step)

    # PutOnHair -> put on hair
    @staticmethod
    def _clean_task(text):
        result = ""
        for char in text:
            if char.isupper():
                result += " " + char.lower()
            else:
                result += char
        result = result.replace(" t v", " TV")
        result = result.replace(" c d", " CD")
        result = result.replace("s i m", "SIM")
        result = result.replace("n b a", "NBA")
        result = result.replace("s s d", "SSD")
        result = result.replace("r j45", "RJ45")
        return result.strip()

    def __len__(self):
        return len(self.annos)

    @staticmethod
    def _create_file_map(video_path):
        file_map = dict()
        files = glob.glob(os.path.join(video_path, "*"))

        for f in files:
            video_id = os.path.splitext(os.path.basename(f))[0]
            if video_id not in file_map:
                file_map[video_id] = f
        return file_map
