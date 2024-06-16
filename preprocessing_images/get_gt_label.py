from datetime import timedelta
import numpy as np
import pandas as pd
import random
import subprocess
### importing functions

from get_gt_label_functions import convert_txt_to_df,change_label,Get_gt_labels,get_txt_from_sentence_dataframe
from cut_videos_functions import extract_clips

# file_path_for_sentences = '/home/summy/Tesis/dataset/manejar_conflictos/annotations/manejar_conflictos_version_sign_nosign_sentence.txt'
# file_path_for_sentences = '/media/iot/ML/Summy/video_cutter/annotations/manejar_conflictos_version_sign_nosign_sentence.txt'



# file_path_for_sentences = '/home/summy/Tesis/dataset/manejar_conflictos/annotations/manejar_conflictos_version_modificado_summy.txt'

file_path_for_words = '/media/iot/ML/Summy/video_cutter/annotations/manejar_conflictos_version_modificado_AGREGADO.txt'


# DataFrame_of_sentences = convert_txt_to_df(file_path_for_sentences) # converted to dataframe and aldo in the delta format
DataFrame_of_words     = convert_txt_to_df(file_path_for_words)    # converted to dataframe and aldo in the delta format

# print(DataFrame_of_words)

vector_to_change=['NN', 'muletilla'] # vector where it is declared the labels to consider as ME

DataFrame_of_words_ME_sign = change_label(DataFrame_of_words,vector_to_change) # Make efective the change of label:
                                                                               #  - NN and muletilla --> turned into ME
                                                                               #  - any other labeles are changed into --> sign

DataFrame_of_words_ME_sign=DataFrame_of_words_ME_sign.drop([1844]) #drop silence

DataFrame_of_sentences=Get_gt_labels(DataFrame_of_words_ME_sign)

print(DataFrame_of_sentences)

output_directory="/media/iot/ML/Summy/video_cutter/cutted_videos/labels"
get_txt_from_sentence_dataframe(DataFrame_of_sentences,DataFrame_of_words_ME_sign,output_directory)


video_file_path='/media/iot/ML/Summy/Data_anotada/manejar_conflictos.mp4'
output_path='/media/iot/ML/Summy/video_cutter/cutted_videos/'

# NumberOfClipsToExtract=3
# extract_clips(DataFrame_of_sentences,video_file_path,output_path,NumberOfClipsToExtract)
extract_clips(DataFrame_of_sentences,video_file_path,output_path)


