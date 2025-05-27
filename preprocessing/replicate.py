"""
This file contains the functions used to replicate frames depending on the window size
"""


import numpy as np
import cv2 
import ffmpeg
import subprocess


def replicate_frames(frame, video_to_write, n, window, total_frames,count): 
    """
    Description:
    ---
    This function pads every video to match the window size for the end and start frame 
    for a 21 size window (DiffAct)

    Input:
    ---

    - frame: frame of an object video
    - video_to_write: video to write on the replicated frames
    - n: number of copies for every frame
    - window: window for feature extraction
    - total_frames: total frames in the video
    - count: track of the number of frame received by the function 

    Output:
    ---
    None--> video saved in path_output

    """

    populate_start=window//2
    populate_end=(window//2) +1 # must be one more for extraction


    if count==0:
            number_repetitions=populate_start+n
    elif count==(total_frames-1):

            number_repetitions=populate_end+n
    else: 
             number_repetitions=n

        
    for _ in range(number_repetitions):
            video_to_write.write(frame)
        

def replicate_frames_for_w16(frame, video_to_write, n, window, total_frames,count): 
    """
    Description:
    ---
    This function pads every video to match the window size for the end and start frame 
    for a 16 size window (MS-TCN)

    Input:
    ---

    - frame: frame of an object video
    - video_to_write: video to write on the replicated frames
    - n: number of copies per frame
    - window: window for feature extraction
    - total_frames: total frames in the video
    - count: track of the number of frame received by the function 

    Output:
    ---
    None--> video saved in path_output

    """

    populate_start=(window//2) +1 # must be one more for extraction
    populate_end=(window//2) 
    

    if count==0:
            number_repetitions=populate_start+n
    elif count==(total_frames-1):

            number_repetitions=populate_end+n
    else: 
             number_repetitions=n

        
    for _ in range(number_repetitions):
            video_to_write.write(frame)
