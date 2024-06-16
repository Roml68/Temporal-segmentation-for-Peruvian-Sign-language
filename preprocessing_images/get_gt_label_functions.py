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

  TimeDeltaFormat = timedelta(seconds=int(seconds), milliseconds=int((milliseconds)), minutes=int(minutes), hours=int(hours))

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
    print("-------------------------------------------------------------------------")

    print("total annotated: "+ " "+ str(len(Dataframe)))
    count_me = Dataframe['label'].value_counts().get('ME', 0)
    print("there are " + str(count_me)+ " "+ 'ME')
    count_sign = Dataframe['label'].value_counts().get('sign', 0)
    print("there are " + str(count_sign)+ " "+ 'sign')

    print("-------------------------------------------------------------------------")

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
    print("total ME identified plus blank spaces " + str(count_me_final)+ " "+ 'ME')
    print("-------------------------------------------------------------------------")
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

def merge_labels(Dataframe, DataFrame_of_sentences_1, label_to_merge):

  """
  This function merges two adjacent labels that are the same and are in the same sentence
  ...

  Input
  -----
  Dataframe: df where the changes are going to be applied
  label_to_merge: label that need to me merged

  Output:
  -------
  df2: df with the indicated label merged

  Example:
  --------

  df_output = merge_labels(df, 'ME')

df:

   |  | label      | start_time   | end_time    |
   |  |----------- |--------------|-------------|
   | 0|sign        |	00:00:30.078|	00:00:32.174|
   | 1|ME          |	00:00:32.174|	00:00:34.078|
   | 1|ME          |	00:00:34.078|	00:00:35.174|

df_output:

   |  | label      | start_time   | end_time    |
   |  |----------- |--------------|-------------|
   | 0|sign        |	00:00:30.078|	00:00:32.174|
   | 1|ME          |	00:00:32.174|	00:00:35.174|

  """
  df2 = Dataframe.copy()
  i=0
  j=0
  sentence_number_temp=-2
  sentence_number_1=-1
  while i < len(df2) - 1:
    j=0
    sentence_number_temp=-2
    sentence_number_1=-1
    if df2.iloc[i]['label'] == label_to_merge :

      j = i + 1
      sentence_number_1 = search_for_sentence_number_given_time(df2.iloc[i],DataFrame_of_sentences_1)
      sentence_number_temp = search_for_sentence_number_given_time(df2.iloc[j],DataFrame_of_sentences_1)

      while (j < len(df2)) and (df2.iloc[j]['label'] == label_to_merge) and (sentence_number_1==sentence_number_temp):
            j += 1
            if j < len(df2):
                    sentence_number_temp = search_for_sentence_number_given_time(df2.iloc[j], DataFrame_of_sentences_1)
      df2.loc[i, 'end_time'] = df2.iloc[j-1]['end_time']
      df2 = pd.concat([df2.iloc[:i+1], df2.iloc[j:]], ignore_index=True)
    i+=1
  count = df2['label'].value_counts().get(label_to_merge, 0)
  print("--------------------------------------------------------------------")
  print("new number of " + label_to_merge + " :"+ str(count) + " when merged" )
  print("--------------------------------------------------------------------")

  return df2





  ####################################################################

def Get_gt_labels(DataFrame_of_words, min_words=7,max_words=14,frame_rate=29.97002997002997):

  """ gets the ground truth label for every frame that contains a sentence in sign language
  ...
  Input
  -----
  DataFrame_of_sentences: Df that contains the start and end time of every sentence
  DataFrame_of_words    : Df that contains the start and end time of every word inside the sentences
  frame_rate            : Frame_rate of the original video

  Output
  -----

  gt: Df of ground truth labels for every frame of the sentence
  """


  gt = pd.DataFrame(columns=['start_time', 'end_time','number_of_words']) # creates the gt Df


  number_of_words=0
  start_time_temp=0
  end_time_temp=0


  while number_of_words < len(DataFrame_of_words): # iteration over every sentence inside the sentence Df
      
      if number_of_words==0:
          start_time_temp=DataFrame_of_words.iloc[number_of_words]['start_time']
          
      else:
          start_time_temp=DataFrame_of_words.iloc[number_of_words]['end_time']
          number_of_words=number_of_words+1


      number_of_selected_words=random.randint(min_words,max_words)
      count_of_signs=0

      for words in range(number_of_words,len(DataFrame_of_words)):
          
        if DataFrame_of_words.iloc[words]['label']=='sign':
            
            count_of_signs=count_of_signs+1
        
            if count_of_signs==number_of_selected_words:
                
                break
            
            elif count_of_signs!=number_of_selected_words and words==len(DataFrame_of_words)-1:
                
                break
            

      number_of_words=words


      if number_of_words>=len(DataFrame_of_words)-1:
          
          number_of_words=len(DataFrame_of_words)-1
          end_time_temp=DataFrame_of_words.iloc[number_of_words]['end_time']


          new_row = { 'start_time': [start_time_temp],
                  'end_time': [end_time_temp],
                  'number_of_words': [number_of_selected_words]}
     

          Dataframe_to_add_data = pd.DataFrame(data=new_row)

          gt = pd.concat([gt, Dataframe_to_add_data], ignore_index = True)
          break

      end_time_temp=DataFrame_of_words.iloc[number_of_words]['end_time']


      new_row = { 'start_time': [start_time_temp],
                  'end_time': [end_time_temp],
                  'number_of_words': [number_of_selected_words]}
     

      Dataframe_to_add_data = pd.DataFrame(data=new_row)

      gt = pd.concat([gt, Dataframe_to_add_data], ignore_index = True)

  return gt

def get_txt_from_sentence_dataframe(DataFrame_of_sentences,DataFrame_of_words,output_directory):
    frame_rate=29.97002997002997


    with open(os.path.join(output_directory,"list_of_labels.txt"), "w") as file:

        for i, sentence_row in DataFrame_of_sentences.iterrows(): # iteration over every sentence inside the sentence Df
          file.write(str(i)+ "\n")


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

          # print(words_within_sentence)

          ################## new


          with open(os.path.join(output_directory,str(i),".txt"), "w") as file:

            for number_of_word_within_sentence in range(0,words_within_sentence.shape[0]):
              number_of_frames_inside_word=0

              start_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].start_time.total_seconds()*frame_rate)/frame_rate))
              end_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].end_time.total_seconds()*frame_rate)/frame_rate))
              end_frame_word = round(end_time_of_the_word_rounded.total_seconds()*frame_rate)
              start_frame_word = round(start_time_of_the_word_rounded.total_seconds()*frame_rate)

              number_of_frames_inside_word=(end_frame_word - start_frame_word)


              for i1 in range(number_of_frames_inside_word):
                  file.write(str(words_within_sentence.iloc[number_of_word_within_sentence].label)+ "\n")

