## Pre-requisities: run 'pip install youtube-dl' to install the youtube-dl package.
## Specify your location of output videos and input json file.
import json
import os
import pathlib

output_path = "./video"
json_path = "./COIN.json"
downloaded_path = "/mnt/nimble/dibyadip/coin/videos"

videos_downloaded = {
    pathlib.Path(line.strip()).stem: line.strip()
    for line in os.listdir(downloaded_path)
}

if not os.path.exists(output_path):
    os.mkdir(output_path)

data = json.load(open(json_path, "r"))["database"]
youtube_ids = set(list(data.keys()))

print(len(videos_downloaded), len(youtube_ids))
exit()
to_download = youtube_ids.difference(videos_downloaded)

for youtube_id in data:
    info = data[youtube_id]
    type = info["recipe_type"]
    url = info["video_url"]
    # vid_loc = output_path + "/" + str(type)
    print(youtube_id)
    exit()
    os.system(
        "youtube-dl -o " + output_path + "/" + youtube_id + ".mp4" + " -f best " + url
    )

    # To save disk space, you could download the best format available
    # 	but not better that 480p or any other qualities optinally
    # See https://askubuntu.com/questions/486297/how-to-select-video-quality-from-youtube-dl
