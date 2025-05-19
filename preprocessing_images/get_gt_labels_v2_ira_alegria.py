from datetime import timedelta
import numpy as np
import pandas as pd
import random
import subprocess
import os

### importing functions

from get_gt_label_functions import convert_txt_to_df,change_label,Get_gt_labels,get_txt_from_sentence_dataframe,convert_to_Delta_format
from cut_videos_functions import extract_clips




def get_Dataframe_of_sentences(dataframe_of_words,min_words=7,max_words=14,small_possible_number_of_words=4,silencio_label=None):


    accumulator_number_of_words_inside_sentence=0

    start_time_temp=dataframe_of_words.iloc[0]['start_time']





    random_number=random.randint(min_words,max_words)

    Dataframe_of_sentences = pd.DataFrame(columns=['start_time', 'end_time','number_of_words'])

    for index, row in dataframe_of_words.iterrows():

        # print(index)

        

        if row['label'] not in ['NN', 'muletilla']:


        


            accumulator_number_of_words_inside_sentence+=1
                    
            

            if accumulator_number_of_words_inside_sentence==random_number:


                new_row = { 'start_time': start_time_temp,
                            'end_time': row['end_time'],
                            'number_of_words': int(accumulator_number_of_words_inside_sentence)}
                    
                start_time_temp=row['end_time']

                accumulator_number_of_words_inside_sentence=0


                    ###3
                Dataframe_to_add_data = pd.DataFrame([new_row])

                Dataframe_of_sentences = pd.concat([Dataframe_of_sentences, Dataframe_to_add_data], ignore_index = True)

                random_number=random.randint(min_words,max_words)
            
                if  row['label']==silencio_label :

                    print("SILENCIOOOOOO")


                    if accumulator_number_of_words_inside_sentence<=small_possible_number_of_words:
                    
                            end_time_temp=row['start_time']
                            Dataframe_of_sentences.loc[Dataframe_of_sentences.index[-1], 'end_time'] = end_time_temp

                    elif accumulator_number_of_words_inside_sentence==random_number or accumulator_number_of_words_inside_sentence>small_possible_number_of_words:
                            new_row = { 'start_time': start_time_temp,
                            'end_time': row['start_time'],
                            'number_of_words': int(accumulator_number_of_words_inside_sentence)}
                        
                            Dataframe_to_add_data = pd.DataFrame([new_row])

                            Dataframe_of_sentences = pd.concat([Dataframe_of_sentences, Dataframe_to_add_data], ignore_index = True)

            


            
            elif row['label']==silencio_label :

                print("SILENCIOOOOOO")


                if accumulator_number_of_words_inside_sentence<=small_possible_number_of_words:
                
                        end_time_temp=row['start_time']
                        Dataframe_of_sentences.loc[Dataframe_of_sentences.index[-1], 'end_time'] = end_time_temp

                elif accumulator_number_of_words_inside_sentence>small_possible_number_of_words:
                        new_row = { 'start_time': start_time_temp,
                        'end_time': row['start_time'],
                        'number_of_words': int(accumulator_number_of_words_inside_sentence)}
                    
                        Dataframe_to_add_data = pd.DataFrame([new_row])

                        Dataframe_of_sentences = pd.concat([Dataframe_of_sentences, Dataframe_to_add_data], ignore_index = True)




                        

                elif accumulator_number_of_words_inside_sentence==1:

                        start_time_temp=row['end_time']

            

                start_time_temp=row['end_time']
                accumulator_number_of_words_inside_sentence=0


            elif index==(len(dataframe_of_words)-1) and accumulator_number_of_words_inside_sentence<random_number: 
                

            
                if accumulator_number_of_words_inside_sentence <= small_possible_number_of_words: # if the end of dataframe


                    Dataframe_of_sentences.at[Dataframe_of_sentences.index[-1], 'end_time'] = row['end_time']

                else: 

                    new_row = { 'start_time': start_time_temp,
                        'end_time': row['end_time'],
                        'number_of_words': int(accumulator_number_of_words_inside_sentence)}
                    
                    Dataframe_to_add_data = pd.DataFrame([new_row])

                    Dataframe_of_sentences = pd.concat([Dataframe_of_sentences, Dataframe_to_add_data], ignore_index = True)
                
                # print("ENDDDDDDDD")
                # print(index)
                # print(accumulator_number_of_words_inside_sentence)
                # print(row['end_time'])
    
    return Dataframe_of_sentences

### stablishing directories

root_directory="/home/summy/Tesis/dataset/"
dataset="ira_alegria"
selected_annotation_file="ira_alegria_anotacion_esteban_ver_final_muletillas.txt"

split_sentence="split_sentence_"+str(0)

file_path_for_words = os.path.join(root_directory,dataset,"annotations",selected_annotation_file)
video_file_path=os.path.join(root_directory,dataset,"raw_data",dataset+".mp4")


output_path_videos=os.path.join(root_directory,dataset,"videos",split_sentence)
output_path_labels=os.path.join(root_directory,dataset,"labels",split_sentence)


dataframe_of_words=convert_txt_to_df(file_path_for_words)




Dataframe_of_sentences=get_Dataframe_of_sentences(dataframe_of_words,min_words=7,max_words=14,small_possible_number_of_words=4,silencio_label="silencio_no")



vector_to_change=['ME', 'descanso'] # vector where it is declared the labels to consider as ME


DataFrame_of_words_ME_sign = change_label(dataframe_of_words,vector_to_change)

#save_dataframe_of_sentences

Dataframe_of_sentences.to_csv(os.path.join(output_path_labels,'Dataframe_of_sentences.txt'), sep=' ', index=True, header=True)

print(Dataframe_of_sentences[100:120])


get_txt_from_sentence_dataframe(Dataframe_of_sentences,DataFrame_of_words_ME_sign,output_path_labels) #getting label file for every sentence

extract_clips(Dataframe_of_sentences,video_file_path,output_path_videos) # cutting the video according to the sentence












    