"""

This file contains utility functions for preprocessing annotation data
into frame-wise labels suitable for machine learning models. It ensures 
that each annotation is accurately mapped to every corresponding frame 
in a given video, enabling supervised learning on frame-level tasks such as 
temporal segmentation or classification.

"""

# Import libraries

from datetime import timedelta
import numpy as np
import pandas as pd
import random
import os


def convert_to_Delta_format(Time):

  """
  Converts time in string format to Delta format, so it is easy to manipulate it
  ...
  Input
  -----
  Time: Time that needs to be converted

  Output
  -----
  TimeDeltaFormat: Time converted

  """


  NumberofSectionsOfTheDate=Time.split(':')

  if len(NumberofSectionsOfTheDate) !=3 :
            raise ValueError("Invalid timestamp format. Should be hh:mm:ss.ms")
  else:
            hours,minutes,seconds_milliseconds = Time.split(':')
            seconds,milliseconds = seconds_milliseconds.split('.')

  TimeDeltaFormat = timedelta(seconds=int(seconds), milliseconds=int(milliseconds), minutes=int(minutes), hours=int(hours))

  return TimeDeltaFormat

  ####################################################################

def convert_txt_to_df(file_path):
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

    # Open the file in read mode
    with open(file_path, "r") as file:
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

  ####################################################################

def change_label(Dataframe, vector_to_change):

    """
    Function that change the labels of the annotated data

    this version considers that there are only two classes so:
      - the labelled words such as: familia, amigo, etc. --> are labelled as 'sign',
      - the labels inside vector_to_change --> are labelled as 'ME'
      - the blank spaces between signs --> are also labelled as 'ME'
    ...
    Input
    -----
    Dataframe: df where the changes are applied
    vector_to_change: labels to turn into ME

    Output
    ------

    df2: dataframe with the implemented changes

    Examples
    --------
    vector_to_change = ['NN']
    df_output = change_label(df, vector_to_change)

    df:

    |  | label      | start_time   | end_time    |
    |  |----------- |--------------|-------------|
    | 0|Hola a todos|	00:00:30.078|	00:00:32.174|
    | 1|NN          |	00:00:34.078|	00:00:35.174|

    df_output:

    |  | label      | start_time   | end_time    |
    |  |----------- |--------------|-------------|
    | 0|sign        |	00:00:30.078|	00:00:32.174|
    | 1|ME          |	00:00:32.174|	00:00:34.078|
    | 2|ME          |	00:00:34.078|	00:00:35.174|




    """
    #identify the number of signs that has the selected labels
    count=0
    count_acumulado=0
    for word in range(len(vector_to_change)):
      count = Dataframe['label'].value_counts().get(vector_to_change[word], 0)
      count_acumulado=count+count_acumulado
      print("there are " + str(count)+ " "+vector_to_change[word])
    print("the total number is : " +" "+ str(count_acumulado))


    # Change the label to 'ME' if it it is inside vector_to_change e.g.'NN' or 'muletilla'

    Dataframe.loc[Dataframe['label'].isin(vector_to_change), 'label'] = 'ME'

    # Change the label to 'sign' for all other labels

    Dataframe.loc[~Dataframe['label'].isin(['ME']), 'label'] = 'sign'

    #print total annotated and modified labels to be ME
    # print("-------------------------------------------------------------------------")

    # print("total annotated: "+ " "+ str(len(Dataframe)))
    # count_me = Dataframe['label'].value_counts().get('ME', 0)
    # print("there are " + str(count_me)+ " "+ 'ME')
    # count_sign = Dataframe['label'].value_counts().get('sign', 0)
    # print("there are " + str(count_sign)+ " "+ 'sign')

    # print("-------------------------------------------------------------------------")

    df2 = Dataframe.copy() #make a copy of the original df

    i=0 #initialize the i variable
    while i < len(df2) - 1: # repeat while i is less than the size of df2

        if df2.iloc[i]['end_time'] != df2.iloc[i+1]['start_time']: # find sections where the end_time from the actual annotation is different from the start_time of the next one
            new_row_data = {'label': 'ME', 'start_time': df2.iloc[i]['end_time'], 'end_time': df2.iloc[i+1]['start_time']} #create a new row that contains the start and end time of the unannotated section
            line = pd.DataFrame([new_row_data]) #convert to df
            df2 = pd.concat([df2.iloc[:i+1], line, df2.iloc[i+1:]], ignore_index=True) #concat df till the actual annotation, the new row and the dataframe from the annotation next to the end

            i += 2 # skip the new created row
        else:
            i += 1 # go to the next annotation

    count_me_final = df2['label'].value_counts().get('ME', 0) # get the number of blank spaces labelled as ME
    # print("total ME identified plus blank spaces " + str(count_me_final)+ " "+ 'ME')
    # print("-------------------------------------------------------------------------")
    return df2

  ####################################################################

def search_for_sentence_number_given_time(word,DataFrame_of_sentences_1):
  i=0
  number_of_sentence=0
  start_time = word['start_time']
  end_time = word['end_time']
  while i < len(DataFrame_of_sentences_1):

    if DataFrame_of_sentences_1.loc[i,'start_time' ]<= start_time and DataFrame_of_sentences_1.loc[i,'end_time' ]>= end_time:
      number_of_sentence=i
      break
    i += 1
  return number_of_sentence


  ####################################################################

def get_Dataframe_of_sentences(dataframe_of_words,min_words=7,max_words=14,small_possible_number_of_words=4,silencio_label=None):
    
    """
    Segments a DataFrame of word-level annotations into sentence-like units using random word count thresholds 
    and silence labels. Skips filler words ('NN', 'muletilla') and groups valid words into sentences until 
    either a random sentence length is reached or a 'silencio' label forces a break, considering minimum 
    sentence size constraints.

    Inputs:
    -------
    - dataframe_of_words : pandas.DataFrame  
      DataFrame with columns:  
        - 'start_time': timestamp in delta format, start time of each word  
        - 'end_time'  : timestamp in delta format, end time of each word  
        - 'label'     : label of each word (e.g., 'persona', 'silencio', 'NN', etc.)

    - min_words : int, optional (default=7)  
      Minimum number of words in a sentence.

    - max_words : int, optional (default=14)  
      Maximum number of words in a sentence.

    - small_possible_number_of_words : int, optional (default=4)  
      Minimum acceptable sentence size when interrupted by silence.

    Output:
    -------
    - pandas.DataFrame  
      New DataFrame with sentence-like segments containing columns:  
        - 'start_time'      : timestamp in delta format, sentence start time  
        - 'end_time'        : timestamp in delta format, sentence end time  
        - 'number_of_words' : int, count of words in the sentence

    
    
    """


    accumulator_number_of_words_inside_sentence=0

    start_time_temp=dataframe_of_words.iloc[0]['start_time']

    random_number=random.randint(min_words,max_words)

    Dataframe_of_sentences = pd.DataFrame(columns=['start_time', 'end_time','number_of_words'])

    for index, row in dataframe_of_words.iterrows():

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
                
    
    return Dataframe_of_sentences
  ####################################################################


def get_txt_from_sentence_dataframe(DataFrame_of_sentences,DataFrame_of_words,output_directory):
    
    """
    Generates frame-level label `.txt` files for each segmented sentence based on word-level annotations. 
    For every sentence in the input DataFrame, it creates a corresponding label file where each frame 
    within that sentence duration is labeled with the associated word's label. Also creates a `list_of_labels.txt` 
    file listing the names of all generated label files.

    Inputs:
    -------
    - DataFrame_of_sentences 
    - DataFrame_of_words 
    - output_directory : Path to the output directory

    Output:
    -------
    - Text files in the specified `output_directory`, one per sentence, containing frame-level labels.
    - A `list_of_labels.txt` file listing all generated `.txt` files, used as an index.

    """
    frame_rate=29.97002997002997


    with open(os.path.join(output_directory.replace('labelsrefined',""),"list_of_labels.txt"), "w") as list_of_labels:

        for i, sentence_row in DataFrame_of_sentences.iterrows(): # iteration over every sentence inside the sentence Df
          list_of_labels.write(str(i)+".txt"+ "\n")


          #iniciating variables

          words_within_sentence = []
          temporal_vector_label = []

          number_of_frames_inside_word = 0
          number_of_frames_inside_word_accumulated = 0
          number_of_frames_inside_sentence = 0


          #---------------------------------------------------------------------
          #getting the start and end time of the sentence
          start_time_of_the_sentence = DataFrame_of_sentences.iloc[i]['start_time']
          end_time_of_the_sentence = DataFrame_of_sentences.iloc[i]['end_time']

          #---------------------------------------------------------------------
          #getting the words that are inside a sentence with a tolerance of 50 ms

          words_within_sentence = DataFrame_of_words[
                    ((DataFrame_of_words['start_time'] >= start_time_of_the_sentence))   &
                    ((DataFrame_of_words['end_time']<= end_time_of_the_sentence))]
          start_time_of_the_sentence_new1 = words_within_sentence.iloc[0]['start_time']
          end_time_of_the_sentence_new1   = words_within_sentence.iloc[-1]['end_time']

          # print(words_within_sentence) # for debugging-->getting the words that are being considered in every sentence

          ################## new


          with open(os.path.join(output_directory,str(i)+".txt"), "w") as file:

            for number_of_word_within_sentence in range(0,words_within_sentence.shape[0]):
              number_of_frames_inside_word=0

              start_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].start_time.total_seconds()*frame_rate)/frame_rate))
              end_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].end_time.total_seconds()*frame_rate)/frame_rate))
              end_frame_word = round(end_time_of_the_word_rounded.total_seconds()*frame_rate)
              start_frame_word = round(start_time_of_the_word_rounded.total_seconds()*frame_rate)

              number_of_frames_inside_word=(end_frame_word - start_frame_word)


              for i1 in range(number_of_frames_inside_word):
                  file.write(str(words_within_sentence.iloc[number_of_word_within_sentence].label)+ "\n")

