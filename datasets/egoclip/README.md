# EgoClip Dataset Preparation
This README describes how to set up the `egoclip/` dataset folder used by the repo.

## Directory layout

```text
egoclip/
├── video_chunks/
└── helping_hands/
    ├── metadata/
    └── hand_object_clip_per_video_4f_lavila_narrator_640/
```

### 1. Download Helping-Hands Annotations
Download the following from [Helping-Hands](https://github.com/Chuhanxx/helping_hand_for_egocentric_videos):
- [`hand_object_clip_per_video_4f_lavila_narrator_640`](https://www.robots.ox.ac.uk/~czhang/hand_object_clip_per_video_4f_lavila_narrator_640.zip)  
- [`metadata`](https://www.robots.ox.ac.uk/~czhang/metadata.zip)

Extract the zips and place both directories inside `egoclip/helping_hands/` as shown above.

### 2. Download Video Chunks
Follow the instructions provided in the [Helping-Hands](https://github.com/Chuhanxx/helping_hand_for_egocentric_videos) repository to download the `video_chunks` used in EgoClip.  

Ensure all paths match the directory layout exactly to allow scripts to locate video and annotation files correctly.