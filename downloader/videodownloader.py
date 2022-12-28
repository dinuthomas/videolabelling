import numpy as np
import pandas as pd
import time
import os
import argparse
import json
import youtube_dl

parser = argparse.ArgumentParser()

parser.add_argument('url', help='video url', type=str)
parser.add_argument('video_folder', help='video download folder', type=str)
parser.add_argument('config', help='config.json', type=str)

args = parser.parse_args()

json_file = args.config
print(json_file)
f = open(json_file)
data = json.load(f)
f.close()

print(data)
max_videoLen = data['videoLength']
duration = data['duration']
videoFolder = data['videoFolder']

url = args.url

def download_movie(url, videoFolder, dur):

    if not os.path.exists(videoFolder):
        os.mkdir(videoFolder)

    videoName = str(os.path.basename(url))+'.mp4'
    videoFile = os.path.join(videoFolder, videoName)

    if os.path.exists(videoFile):
        print("video file already present")
        return videoFile

    st_time = "00:00:05.00"

    shell_cmd = 'youtube-dl --external-downloader ffmpeg --external-downloader-args '+'"'+' -ss '+str(st_time)+' -t '+str(dur)+'"'+' -f best '+str(url)+' -o '+str(videoFile)

    os.system(shell_cmd)
    return videoFile

print("start download")

start_time = time.time()
videoFile = download_movie(url, videoFolder, duration)
