import numpy as np
import cv2 
import ffmpeg
import subprocess

#replicar cada frame y las anotaciones

"""
Description:
---
This function replicates every frame of a video n times and adds 
the frames to compensate the window in the feature extraction phase

Input:
---

- path: path of the input video
- path_output: path of the output video MUST INCLUDE .mp4
- n: number of copies for every frame
- window: window for feature extraction

Output:
---
None--> video saved in path_output

"""

def FrameCapture(original_video, video_to_write, n, window): 

    populate_start=window//2
    populate_end=(window//2) +1 # must be one more for extraction
    

    count=0
    # vidObj object calls read 
    # function extract frames 
    success, image = original_video.read() 
        # frame_copied=image.copy()

    if count==0:
            number_repetitions=populate_start+n
    elif count==(total_frames-1):

            number_repetitions=populate_end+n
    else: 
             number_repetitions=n

        
    for _ in range(number_repetitions):
            video_to_write.write(image)



  
        # # Saves the frames with frame-count 
        # cv2.imwrite("frame%d.jpg" % count, image) 
        # cv2.imwrite("framecop%d.jpg" % count, image) 

  
    count += 1
        

    
def Label_repetition(label_path,output_path,n):

    with open(label_path, "r") as file:
        lines = file.readlines()

    labels=[line.strip() for line in lines]

    total_lines=len(labels)
    count_lines=0
    
    while count_lines < total_lines:
        with open(output_path, "a") as outfile:
            for _ in range(n):
                    outfile.write(labels[count_lines] + '\n')
            
        count_lines=count_lines+1


    
# path="/home/summy/Tesis/dataset/manejar_conflictos/labels/label_names_0.txt"
# n = 2  # Number of copies of each frame
# populate_start=10
# populate_end=11
# output_path = "/home/summy/Tesis/dataset/manejar_conflictos/labels/post_labels.txt"


# Label_repetition(path,output_path,n)