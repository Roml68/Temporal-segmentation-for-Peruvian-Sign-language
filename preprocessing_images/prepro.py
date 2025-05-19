import numpy as np
import cv2 
import ffmpeg
import subprocess
import matplotlib.pyplot as plt
import os

#### Importing functions

from replicate import replicate_frames, replicate_frames_for_w16
from replicate import Label_repetition
from equa1 import corlorCorrection_and_histequalization


#### Getting the directories

root_dir="/home/summy/Tesis/"

# selected_database="manejar_conflictos"
# selected_database="ira_alegria"
selected_database="test_data"



root_database=os.path.join(root_dir,"dataset",selected_database)  
videos=os.path.join(root_database,"videos","split_sentence_0")
list_of_videos=os.path.join(root_database,"list_of_videos_original.txt")


output_directory_videos="preprocessed_videos/set_21_onlyreplicated"

#### Getting the paths of the videos 

with open(list_of_videos, "r") as videofile:
    paths = videofile.readlines()

# Strip newline characters from each path

paths = [path.strip() for path in paths]
paths = [os.path.join(videos,path) for path in paths]

### Generating variables
count_files=0
number_of_copies=1
window= 21 #15 # --> 16 #21 #it is 15 just for the function

### Applying the functions to preprocess

while count_files < len(paths): 


    #getting the names of the output files.mp4

    output_path=os.path.join(root_database,output_directory_videos,str(count_files)+".mp4")

    #getting information from the original video

    original_video = cv2.VideoCapture(paths[count_files]) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = original_video.get(cv2.CAP_PROP_FPS)
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))


    print("initial frame count", total_frames)

    #creating the object to write the new video

    video_to_write = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
  
    
    count_frames=0 #stores the actual frame number to preprocess

    ### Going inside every frame to preprocess

    while count_frames < total_frames:  

            _,original_frame = original_video.read() 
            # new_frame=corlorCorrection_and_histequalization(original_frame,lower_bound=40,upper_bound=220,color_domain="hsv", apply_filtering=False)

            replicate_frames(original_frame, video_to_write, number_of_copies, window, total_frames,count_frames)
            # replicate_frames_for_w16(original_frame, video_to_write, number_of_copies, window, total_frames,count_frames)


            count_frames=count_frames+1

    # closing the original and the generated video
    original_video.release()
    video_to_write.release() 
    cv2.destroyAllWindows()

    count_files=count_files+1 # stores the actual number of video being processed
