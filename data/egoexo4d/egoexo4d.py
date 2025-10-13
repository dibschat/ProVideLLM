import glob, json, math, os, re, torch, tqdm
from functools import reduce

from transformers import pipeline


class EGOEXO4D:
    root = "datasets/egoexo4d/"
    video_root = os.path.join(root, "videos")
    anno_root = os.path.join(root, "instructions")

    def __init__(
        self,
        split: str,
        vision_pretrained: str,
        dataset_dir: str,
        frame_fps: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = os.path.join(dataset_dir, "egoexo4d")

        self.video_root = os.path.join(self.root, "videos")
        self.anno_root = os.path.join(self.root, "instructions")

        self.frame_fps = frame_fps

        assert split in ["train", "val", "trainval", "test"]
        self.split = split

        annos = json.load(
            open(os.path.join(self.anno_root, f"instructions_{self.split}.json"))
        )
        file_map = self._create_file_map(self.anno_root, self.video_root)

        self._annos = [
            {
                "index": anno["id"],
                "video_uid": anno["video_id"],
                "video_name": anno["video_name"],
                "step": self._clean_step(anno["answer"]),
                "question": anno["question"],
                "start_frame": anno["start_frame"],
                "end_frame": anno["end_frame"],
                "answer_id": anno["answer_id"],
                "video_path": file_map[anno["video_id"]],
            }
            for anno in annos
            if ("train" in self.split and anno["video_id"] in file_map)
            or "train" not in self.split
        ]

        self.step_categories = [
            self._clean_step(line.strip())
            for line in open(os.path.join(self.anno_root, f"steps.txt"), "r")
        ]

    def __len__(self):
        return len(self.annos)

    # remove . at the end, covid- 19/covid 19 --> covid 19
    @staticmethod
    def _clean_step(step):
        return re.sub(r"covid- 19", "covid 19", re.sub(r"\.$", "", step))

    @staticmethod
    def _create_file_map(anno_path, video_path):
        trainval_meta = json.load(
            open(os.path.join(anno_path, f"instructions_trainval.json"))
        )
        test_meta = json.load(open(os.path.join(anno_path, f"instructions_test.json")))
        metadata = trainval_meta + test_meta
        name_id_map = {meta["video_name"]: meta["video_id"] for meta in metadata}

        file_map = dict()
        files = glob.glob(os.path.join(video_path, "*"))

        for f in files:
            video_id = os.path.basename(f).replace("_rgb", "").replace(".pt", "")
            if video_id in name_id_map:
                video_id = name_id_map[video_id]
            if video_id not in file_map:
                file_map[video_id] = f
        return file_map
