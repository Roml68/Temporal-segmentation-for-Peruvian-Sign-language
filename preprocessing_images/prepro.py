import numpy as np
import cv2 
import ffmpeg
import subprocess
import matplotlib.pyplot as plt
import os

#### Importing functions

from replicate import FrameCapture
from replicate import Label_repetition


#### Getting the directories

root_dir="/home/summy/Tesis/"

selected_database="manejar_conflictos"

root_database=os.path.join(root_dir,"dataset",selected_database)
videos=os.path.join(root_database,"videos")
labels=os.path.join(root_database,"labels")
list_of_videos=os.path.join(root_database,"list_of_videos_original.txt")
list_of_labels=os.path.join(root_database,"list_of_labels_original.txt")

output_directory_videos="preprocessed_videos"
output_directory_labels="groundTruth"
output_directory_list_videos=os.path.join(root_database,"preprocessed_videos.txt")
output_directory_list_labels=os.path.join(root_database,"preprocessed_labels.txt")



#### Getting the paths of the videos 

with open(list_of_videos, "r") as videofile:
    paths = videofile.readlines()

with open(list_of_labels, "r") as labelfile:
    paths_labels = labelfile.readlines()

# Strip newline characters from each path

paths = [path.strip() for path in paths]
paths = [os.path.join(videos,path) for path in paths]

paths_labels = [path.strip() for path in paths_labels]
paths_labels = [os.path.join(labels,path) for path in paths_labels]

### Generating variables
count=0
number_of_copies=2
window=21

### Applying the functions to preprocess

while count < len(paths): 
    
    output_path=os.path.join(root_database,output_directory_videos,str(count)+".mp4")

    vidObj = cv2.VideoCapture(paths[count]) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = int(vidObj.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print("initial frame count", total_frames)

    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    FrameCapture(paths[count], output_path, number_of_copies, window)


    output_path_labels=os.path.join(root_database,output_directory_labels,str(count)+".txt")
    Label_repetition(paths_labels[count], output_path_labels,number_of_copies)

    with open(output_directory_list_videos,"w") as videofile_output:
        videofile_output.write(str(count)+".mp4")

    with open(output_directory_list_labels,"w") as labelfile_output:
        labelfile_output.write(str(count)+".txt")

        
    

    count=count+1


