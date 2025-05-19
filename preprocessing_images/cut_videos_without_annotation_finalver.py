import random
from datetime import timedelta
import pandas as pd
import os
import cv2

# Convert time strings to seconds
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s).total_seconds()

def seconds_to_time(seconds):
    return timedelta(seconds=seconds)

# Parameters
start_time_video = "00:00:31.600"
end_time_video = "00:25:38.307"
silencios = [["00:02:24.787", "00:04:16.652"], 
             ["00:06:38.627", "00:06:47.987"], 
             ["00:06:58.413", "00:07:08.440"], 
             ["00:15:14.933", "00:15:24.440"]]

# Convert ranges to seconds
start_time_sec = time_to_seconds(start_time_video)
end_time_sec = time_to_seconds(end_time_video)
silencios_sec = [(time_to_seconds(start), time_to_seconds(end)) for start, end in silencios]
min_sec=6
max_sec=10

# Generate video segments while avoiding silence ranges


def get_df_videos_from_unnanottated(start_time_sec,end_time_sec,min_sec,max_sec):

    segments = []
    current_time = start_time_sec

    video_number=0
    while current_time < end_time_sec:
        duration = random.uniform(min_sec, max_sec)
        end_segment = current_time + duration

        # Adjust the segment for silence ranges
        for silence_start, silence_end in silencios_sec:
            if current_time < silence_start < end_segment:  # Ends at silence start
                end_segment = silence_start
            elif silence_start <= current_time < silence_end:  # Starts after silence
                current_time = silence_end
                end_segment = current_time + duration
        
        # Ensure the fragment doesn't exceed video bounds
        if end_segment > end_time_sec:
            end_segment = end_time_sec

        # Add segment and handle small fragments
        if segments and end_segment - current_time < min_sec:
            prev_start, prev_end, name = segments.pop()
            segments.append((prev_start, end_segment, f"{video_number}.mp4"))
        elif end_segment - current_time >= min_sec:
            segments.append((current_time, end_segment, f"{video_number}.mp4"))
        
        current_time = end_segment
        video_number+=1

    # Convert segments to a readable format
    formatted_segments = [(seconds_to_time(start), seconds_to_time(end), name) for start, end, name in segments]

    # Result DataFrame

    df = pd.DataFrame(formatted_segments, columns=["start_time", "end_time", "name_of_video"])



    return df


import subprocess
from datetime import timedelta

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
    # pixels where the signer is located

    #varlores originales
    # y1 = 380
    # y2 = 600
    # x1 = 988
    # x2 = 1208

    #modificado

    y1 = 400
    y2 = 630
    x1 = 1015
    x2 = 1220


    TimeDeltaFormat_start = timedelta(seconds=(round(start_time.total_seconds()*frame_rate)/frame_rate))
    TimeDeltaFormat_end   = timedelta(seconds=(round(end_time.total_seconds()*frame_rate)/frame_rate))

    print(TimeDeltaFormat_start)
    print(TimeDeltaFormat_end)
    print(round(TimeDeltaFormat_start.total_seconds()*frame_rate))
    print(round(TimeDeltaFormat_end.total_seconds()*frame_rate))

    start_frame=round(TimeDeltaFormat_start.total_seconds()*frame_rate)
    end_frame=round(TimeDeltaFormat_end.total_seconds()*frame_rate)


    # ffmpeg_command = [
    #     "ffmpeg",
    #     "-i", input_file_path,
    #     "-filter_complex",
    #     f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS,crop={x2 - x1}:{y2 - y1}:{x1}:{y1}[v]",
    #     "-r", str(frame_rate),
    #     "-map", "[v]",
    #     output_file_path
    # ]

    ffmpeg_command = [
    "ffmpeg",
    "-i", input_file_path,
    "-filter_complex",
    f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS,crop={x2 - x1}:{y2 - y1}:{x1}:{y1},scale=220:220[v]",
    "-r", str(frame_rate),
    "-map", "[v]",
    output_file_path
]

    # Run the ffmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True)
        print("FFmpeg command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

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



        # NameFileSplittedbyUnderscore= df.label[i].replace(",", "").replace(" ", "_") # gets the sentence label separated by underscore
        # output_file_path = output_path + name_file[len(name_file)-1].replace(".mp4", "") +'/'+ str(NameFileSplittedbyUnderscore + '_'+ str(i) + '.mp4')
        output_file_path = os.path.join(output_path, str(i) + '.mp4')
        get_segment(start_time,end_time,frame_rate,video_file_path,output_file_path)
        print("frames:",start_frame,end_frame)
        print(frame_rate)
        print(output_file_path)
        # file.write("./"+str(NameFileSplittedbyUnderscore + '_'+ str(i) + '.mp4')+ "\n")
        file.write(str(i) + '.mp4'+ "\n")


        i+=1



video_file_path="/home/summy/Tesis/30_04_2020.mp4"

output_path="/home/summy/Tesis/dataset/test_videos"

df=get_df_videos_from_unnanottated(start_time_sec,end_time_sec,min_sec,max_sec)

df.to_csv(os.path.join(output_path,"video_segments.txt"), sep="\t", index=False, header=True)

extract_clips(df,video_file_path,output_path)







print(df[90:100])
