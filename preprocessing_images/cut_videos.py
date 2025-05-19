from datetime import timedelta
import numpy as np
import pandas as pd
import random
### importing functions
from  get_gt_label_functions import change_label,Get_gt_labels
from cut_videos_functions import convert_txt_to_df

file_path_for_sentences = '/home/summy/Tesis/dataset/manejar_conflictos/annotations/manejar_conflictos_version_sign_nosign_sentence.txt'

# file_path_for_words = '/content/drive/MyDrive/Tesis/annotation/ira_alegria_ME_sena_v2.txt'
file_path_for_words = '/home/summy/Tesis/dataset/manejar_conflictos/annotations/manejar_conflictos_version_modificado_summy.txt'


DataFrame_of_sentences = convert_txt_to_df(file_path_for_sentences) # converted to dataframe and aldo in the delta format
DataFrame_of_words     = convert_txt_to_df(file_path_for_words)    # converted to dataframe and aldo in the delta format

# print(DataFrame_of_words)

vector_to_change=['NN', 'muletilla'] # vector where it is declared the labels to consider as ME

DataFrame_of_words_ME_sign = change_label(DataFrame_of_words,vector_to_change) # Make efective the change of label:
                                                                               #  - NN and muletilla --> turned into ME
                                                                               #  - any other labeles are changed into --> sign

gt=Get_gt_labels(DataFrame_of_sentences,DataFrame_of_words)

print(gt)
