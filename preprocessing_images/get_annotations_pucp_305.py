import os
from get_gt_label_functions import convert_to_Delta_format,change_label
import pandas as pd
from datetime import timedelta
import difflib

##### FUNCTIONS ###########

def convert_txt_to_df(dir,file_path):
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
    00:00:30.078 - 00:00:32.174 --> converted in delta format

   |  | label                                | start_time   | end_time    |
   |  |--------------------------------------|--------------|-------------|
   | 0|Hola a todos, bonita mañana para todos|	00:00:30.078|	00:00:32.174|


    """
    start_time = []
    end_time = []
    label=[]

    df = pd.DataFrame()

    # Open the file in read mode
    
    with open(os.path.join(dir,file_path), "r") as file:
            lines = file.readlines()

    i=0

    while i < len(lines):
            label.append(lines[i].strip())
            ini, fin=lines[i+1].strip().split("-")
            ini=ini.strip()
            fin=fin.strip()
            start_time.append(ini)
            end_time.append(fin)

            i += 2

    data = {'label': label, 'start_time': start_time, 'end_time': end_time}

       
    df = pd.DataFrame(data)
        # convert the dataframe to delta format
    df['start_time']=df['start_time'].apply(convert_to_Delta_format)
    df['end_time']=df['end_time'].apply(convert_to_Delta_format)
    return df       


def get_txt_from_sentence_dataframe(index,words_within_sentence,output_directory):
    frame_rate=29.97002997002997

    acc=0

    with open(os.path.join(output_directory,str(index)+".txt"), "w") as file:

            for number_of_word_within_sentence in range(0,words_within_sentence.shape[0]):
              number_of_frames_inside_word=0

              start_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].start_time.total_seconds()*frame_rate)/frame_rate))
              end_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].end_time.total_seconds()*frame_rate)/frame_rate))
              end_frame_word = round(end_time_of_the_word_rounded.total_seconds()*frame_rate)
              start_frame_word = round(start_time_of_the_word_rounded.total_seconds()*frame_rate)

              number_of_frames_inside_word=(end_frame_word - start_frame_word)
              acc=acc+number_of_frames_inside_word
              
             


              for i1 in range(number_of_frames_inside_word):
                  file.write(str(words_within_sentence.iloc[number_of_word_within_sentence].label)+ "\n")

            print(acc)
def get_index_from_list(list_of_videos, target_filename):
    with open(list_of_videos, 'r') as file:
        lines = file.readlines()
    
    # Clean and convert filenames
    modified_lines = [line.strip().replace(".mp4", ".txt") for line in lines]
    
    # Create DataFrame
    df_videos = pd.DataFrame(modified_lines, columns=["video_filenames"])
    
    # Find index of target_filename
    match = df_videos[df_videos["video_filenames"] == target_filename]
    
    if not match.empty:
        return match.index[0]
    else:
        print(target_filename)
        return None  # or "Filename not found"
    
def count_related_fuzzy_from_df(df, keywords, threshold=0.8):
    counts = {keyword: 0 for keyword in keywords}
    
    for label in df['label']:
        label = str(label)  # ensure it's string
        for keyword in keywords:
            similarity = difflib.SequenceMatcher(None, keyword, label).ratio()
            if similarity >= threshold:
                counts[keyword] += 1
    return counts
##############################

root_dir="/home/summy/Tesis"
annotations_dir=os.path.join(root_dir,"PUCP_305_RE","modified_annotations")
list_of_videos_dir=os.path.join(root_dir,"dataset","305_PUCP","list_of_videos_original.txt")
output_dir=os.path.join(root_dir,"PUCP_305_RE","labels")

dir_list = os.listdir(annotations_dir)








# dir_list=dir_list[:2]

vector_to_change=['ME', 'descanso'] # vector where it is declared the labels to consider as ME

categories_to_count=["seña gestual", "NN", "descanso" ,"muletilla"]

total_counts = {category: 0 for category in categories_to_count}


for file_path in dir_list:
       

    df=convert_txt_to_df(annotations_dir,file_path)
    counts=count_related_fuzzy_from_df(df, categories_to_count, threshold=0.6)
    for category, count in counts.items():
            total_counts[category] += count

    # DataFrame_of_words_ME_sign = change_label(df,vector_to_change)
    
    # index=get_index_from_list(list_of_videos_dir,file_path)
    # print(f"-----------{index}----------------")
    # print(DataFrame_of_words_ME_sign)
    # get_txt_from_sentence_dataframe(index,DataFrame_of_words_ME_sign,output_dir)
    


print(total_counts)