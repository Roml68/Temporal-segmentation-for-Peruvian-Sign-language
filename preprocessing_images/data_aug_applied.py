import numpy as np
import cv2 
import ffmpeg
import subprocess
import matplotlib.pyplot as plt
import os
import random

#### Importing functions

from replicate import replicate_frames
from replicate import Label_repetition
from equa1 import corlorCorrection_and_histequalization
from data_augmenation import rotate_image,corlorCorrection,zoom_image,translate_image


#### Getting the directories

root_dir="/home/summy/Tesis/"

selected_database="manejar_conflictos"

split="split_sentence_0"

name_output_directory="translation_x_y_15_15"

root_database=os.path.join(root_dir,"dataset",selected_database)
videos=os.path.join(root_database,"videos",split)
labels=os.path.join(root_database,"labels")
list_of_videos=os.path.join(root_database,"list_of_videos_original.txt")
list_of_labels=os.path.join(root_database,"list_of_labels_original.txt")

output_directory_videos=os.path.join("preprocessed_videos",name_output_directory)
output_directory_labels="groundTruth"
output_directory_list_videos=os.path.join(root_database,"preprocessed_videos.txt")
output_directory_list_labels=os.path.join(root_database,"preprocessed_labels.txt")

### Data augmentation conf

#range of angles --> 0

max_right_rotation= 10
max_left_rotation= -10

#zoomin_zoomout --> 1

zoom_out= 0.8
zoom_in= 1.2  

#translation

min_x_translation=-15
max_x_translation=15

min_y_translation=-15
max_y_translation=15






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
count_files=0
number_of_copies=1
window=21

### Applying the functions to preprocess

while count_files < len(paths): 


    #getting the names of the output files.mp4
    output_path=os.path.join(root_database,output_directory_videos,str(count_files)+".mp4")

    #getting information from the original video
    original_video = cv2.VideoCapture(paths[count_files]) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = int(vidObj.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = original_video.get(cv2.CAP_PROP_FPS)
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))


    print("initial frame count", total_frames)

    #creating the object to write the new video
    video_to_write = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
  
    
    count_frames=0 #stores the actual frame number to preprocess

    ### changing angles

         
    # random_angle=np.random.randint(max_left_rotation,max_right_rotation)

    random_angle=0

    ### zooming in and out

    # random_zoom=random.uniform(zoom_out,zoom_in)    

    random_zoom=1

    ### translate to the right and left

    random_x=np.random.randint(min_x_translation,max_x_translation)
    random_y=np.random.randint(min_y_translation,max_y_translation)





    ### Going inside every frame to preprocess
    while count_frames < total_frames:  

            _,original_frame = original_video.read() 
            # new_frame=corlorCorrection_and_histequalization(original_frame,lower_bound=40,upper_bound=220,color_domain="hsv", apply_filtering=False)
            new_frame=corlorCorrection(original_frame)
            # new_frame=corlorCorrection(original_frame)
            new_frame=rotate_image(new_frame,angle=random_angle)
            new_frame=zoom_image(new_frame, scale=random_zoom)
            new_frame=translate_image(new_frame, x_shift=random_x, y_shift=random_y)


            replicate_frames(new_frame, video_to_write, number_of_copies, window, total_frames,count_frames)

            count_frames=count_frames+1

    # closing the original and the generated video
    original_video.release()
    video_to_write.release() 
    cv2.destroyAllWindows()




    ### Preprocesing the labels so they match the new videos
    # output_path_labels=os.path.join(root_database,output_directory_labels,str(count_files)+".txt")
    # Label_repetition(paths_labels[count_files], output_path_labels,number_of_copies)


    ### Storing the names of the video and label files--> getting a list 
    with open(output_directory_list_videos,"w") as videofile_output:
        videofile_output.write(str(count_files)+".mp4")

    with open(output_directory_list_labels,"w") as labelfile_output:
        labelfile_output.write(str(count_files)+".txt") 



    count_files=count_files+1 # stores the actual number of video being processed
