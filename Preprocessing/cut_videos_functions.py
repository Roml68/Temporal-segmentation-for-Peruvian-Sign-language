"""
This file contains the functions used to cut the video into smaller fragments given a range of time

"""


import numpy as np
import pandas as pd
import cv2
import os
import subprocess
from datetime import timedelta
import subprocess


# Functions
# -------------------------------------------------------------------------------------------------------------------------------------------------------


def convert_txt_to_df(file_path):
    """
    Function converts annotations to a dataframe
    ...
    Input
    -----
    file_path: path to the file

    Output
    ------
    df: Dataframe

    Example
    -------
    Hola a todos, bonita mañana para todos
    00:00:30.078 - 00:00:32.174

   |  | label                                | start_time   | end_time    |
   |  |--------------------------------------|--------------|-------------|
   | 0|Hola a todos, bonita mañana para todos|	00:00:30.078|	00:00:32.174|


    """
    start_time = []
    end_time = []
    label=[]

    # Open the file in read mode
    with open(file_path, "r") as file:
        lines = file.readlines()

    i=0

    while i < len(lines):
      label.append(lines[i].strip())
      ini, fin=lines[i+1].strip().split("-")
      start_time.append(ini)
      end_time.append(fin)

      i += 2

    data = {'label': label, 'start_time': start_time, 'end_time': end_time}
    df = pd.DataFrame(data)
    return df

##############################################################################################################################################################

def count_frames(video_path):

    """
    counts the frame rate
    ...
    input
    -----
    video_path: path of the video that counts the total of frames of a video and calculates the frame rate

    Output
    ------
    total_frames: the total number of frames in the video
    frame_rate: frame rate of the video

    Example
    -------
    I:
    /content/drive/MyDrive/Tesis/annotation/ira_alegria_sentence_v3.txt

    O:
    10000,30


    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return -1

    # Get the total number of frames in the video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return total_frames,frame_rate


def timestamp_to_frame(timestamp, frame_rate=30): #code to get number of the frame when given timestamp format

    """
    code to get the frame number when given the timestamp format
    ...

    Input
    -----
    timestamp: timestamp where the selected frame is located
    frame_rate: frame_rate of the video

    Output
    ------
    frame_number: frame number of the selected timestamp

    Example
    -------
    I:
    00:00:30.078

    O:
    173


    """
    parts=timestamp.split(':')

    if len(parts) != 3:
        raise ValueError("Invalid timestamp format. Should be hh:mm:ss.ms")
    else:
      hours,minutes,seconds_milliseconds = timestamp.split(':')
      seconds,milliseconds = seconds_milliseconds.split('.')

    #seconds, milliseconds = divmod(seconds_milliseconds, 1000)

    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    frame_number = round(total_seconds * frame_rate)
    return frame_number

##############################################################################################################################################################

def get_segment(start_time, end_time, frame_rate, input_file_path, output_file_path):

    """
    cuts a short clip of a long video, considering the frames between two frame numbers
    ...

    Input
    -----
    start_time: time when the clip starts
    end_time: time when the clip ends
    frame_rate: frame_rate of the resulting video
    input_file_path: path of the long video
    output_file_path: path where the clips are going to be saved

    Output
    -----
    None-->saved clips in the output path

    """
    # range of pixels where the signer is located
    y1 = 380
    y2 = 600
    x1 = 988
    x2 = 1208


    TimeDeltaFormat_start = timedelta(seconds=(round(start_time.total_seconds()*frame_rate)/frame_rate))
    TimeDeltaFormat_end   = timedelta(seconds=(round(end_time.total_seconds()*frame_rate)/frame_rate))

    print(TimeDeltaFormat_start)
    print(TimeDeltaFormat_end)
    print(round(TimeDeltaFormat_start.total_seconds()*frame_rate))
    print(round(TimeDeltaFormat_end.total_seconds()*frame_rate))

    start_frame=round(TimeDeltaFormat_start.total_seconds()*frame_rate)
    end_frame=round(TimeDeltaFormat_end.total_seconds()*frame_rate)


    ffmpeg_command = [
        "ffmpeg",
        "-i", input_file_path,
        "-filter_complex",
        f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS,crop={x2 - x1}:{y2 - y1}:{x1}:{y1}[v]",
        "-r", str(29.97002997002997),
        "-map", "[v]",
        output_file_path
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True)
        print("FFmpeg command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
##############################################################################################################################################################

def extract_clips(df,video_file_path,output_path,number_clips=None):

  """
  recurrent funtion that extracts the asked number of clips, according to the annotated sentences
  ...

  Input
  -----
  df: Dataframe iof sentences 
  video_file_path: video containing the signs
  number_clips: number of clips required if None then it gets the same number as the annotated sentences

  Output
  ------
  ---> short clips are saved in the output path <---
  df: dataframe where the anotated data is organized

  """
  i=0
  total_frames,frame_rate=count_frames(video_file_path)

  if number_clips == None:
    number_clips = df.shape[0]

  with open(os.path.join(output_path,"videos_list.txt"), "w") as file:

      while i < number_clips:

        start_time  = df.start_time[i] # gets the start time of the clip
        end_time    = df.end_time[i]   # gets the end time of the clip
        start_frame = round(start_time.total_seconds()*frame_rate) # converts start time to start frame
        end_frame   = round(end_time.total_seconds()*frame_rate) # converts end time to end frame
        name_file   = video_file_path.split('/') # gets an array of every string that contains the input path


        output_file_path = os.path.join(output_path, str(i) + '.mp4')
        get_segment(start_time,end_time,frame_rate,video_file_path,output_file_path)
        print("frames:",start_frame,end_frame)
        print(frame_rate)
        print(output_file_path)
        file.write(str(i) + '.mp4'+ "\n")


        i+=1
