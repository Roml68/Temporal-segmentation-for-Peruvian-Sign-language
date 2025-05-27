"""
This file is used to generate the respective frame-wise annotations for manejar_conflictos dataset
"""


from datetime import timedelta
import numpy as np
import pandas as pd
import random
import subprocess
import os

### importing functions

from get_gt_label_functions import convert_txt_to_df,change_label,Get_gt_labels,get_txt_from_sentence_dataframe,convert_to_Delta_format,get_Dataframe_of_sentences
from cut_videos_functions import extract_clips


### stablishing directories

root_directory="/home/summy/Tesis/dataset/"
dataset="manejar_conflictos"
selected_annotation_file="manejar_conflictos_refinado.txt"

split_sentence="split_sentence_"+str(0)

file_path_for_words = os.path.join(root_directory,dataset,"annotations",selected_annotation_file)
video_file_path=os.path.join(root_directory,dataset,"raw_data",dataset+".mp4")


output_path_videos=os.path.join(root_directory,dataset,"videos",split_sentence)
output_path_labels=os.path.join(root_directory,dataset,"labels",split_sentence)


dataframe_of_words=convert_txt_to_df(file_path_for_words)

Dataframe_of_sentences=get_Dataframe_of_sentences(dataframe_of_words,min_words=7,max_words=14,small_possible_number_of_words=4,silencio_label="silencio")

vector_to_change=['ME', 'descanso'] # vector where it is declared the labels to consider as ME

DataFrame_of_words_ME_sign = change_label(dataframe_of_words,vector_to_change)

Dataframe_of_sentences.to_csv(os.path.join(output_path_labels,'Dataframe_of_sentences.txt'), sep=' ', index=True, header=True)


get_txt_from_sentence_dataframe(Dataframe_of_sentences,DataFrame_of_words_ME_sign,output_path_labels) #getting label file for every sentence

extract_clips(Dataframe_of_sentences,video_file_path,output_path_videos) # cutting the video according to the sentence











    