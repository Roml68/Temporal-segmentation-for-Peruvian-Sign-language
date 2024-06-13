from datetime import timedelta
import numpy as np
import pandas as pd

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

def Get_gt_labels(DataFrame_of_sentences,DataFrame_of_words,frame_rate=29.97002997002997):

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


  gt = pd.DataFrame(columns=['label', 'gt_label_per_word']) # creates the gt Df

  #--------------------------------------------------------------------------------------
  # preliminar initialization fo the accumulative variables

  number_of_frames_inside_word_accumulated_total = 0
  number_of_frames_inside_sentence_total = 0

  number_of_frames_inside_word = 0
  number_of_frames_inside_word_accumulated = 0
  number_of_frames_inside_sentence = 0

  #----------------------------------------------------------------------------------------
  # making sure that every sentence is exaclty next to each other

  i=0
  while i < len(DataFrame_of_sentences) - 1:
      if DataFrame_of_sentences.iloc[i]['end_time'] != DataFrame_of_sentences.iloc[i+1]['start_time']:
         DataFrame_of_sentences.loc[i, 'end_time'] = DataFrame_of_sentences.iloc[i+1]['start_time']
      i+=1
  #-----------------------------------------------------------------------------------------
  # iteration to find the words inside every sentence and get the number of frames
  # - the condition is that there must the same amount of frames inside a sentence as the acummulated number of frames per word inside every sentence

  for i, sentence_row in DataFrame_of_sentences.iterrows(): # iteration over every sentence inside the sentence Df


      number_of_frames_inside_word_accumulated_total=number_of_frames_inside_word_accumulated_total+number_of_frames_inside_word_accumulated
      number_of_frames_inside_sentence_total = number_of_frames_inside_sentence_total + number_of_frames_inside_sentence

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
                 ((DataFrame_of_words['start_time'].apply(lambda x: abs(x - start_time_of_the_sentence).total_seconds()) <= 0.106) |
                  (DataFrame_of_words['start_time'] >= start_time_of_the_sentence))   &
                 ((DataFrame_of_words['end_time'] <= end_time_of_the_sentence) |
                 (DataFrame_of_words['end_time'].apply(lambda x: abs(x - end_time_of_the_sentence).total_seconds()) <= 0.106) ) &
                  (DataFrame_of_words['start_time'] < end_time_of_the_sentence)]

      #---------------------------------------------------------------------
      #adjusting the time of the sentence, so it is exactly the same as the start of the first word and the end of the last word

      start_time_of_the_sentence_new=words_within_sentence.iloc[0]['start_time']
      end_time_of_the_sentence_new =words_within_sentence.iloc[-1]['end_time']

      DataFrame_of_sentences.loc[i, 'start_time'] = start_time_of_the_sentence_new
      DataFrame_of_sentences.loc[i, 'end_time'] = end_time_of_the_sentence_new



      #------------------------------------------------------------------------------
      # rounding the start and end times for the sentence and every word


      start_time_of_the_sentence_rounded = timedelta(seconds=(round(start_time_of_the_sentence_new.total_seconds()*frame_rate)/frame_rate))
      end_time_of_the_sentence_rounded = timedelta(seconds=(round(end_time_of_the_sentence_new.total_seconds()*frame_rate)/frame_rate))
      end_frame_sentence =round(end_time_of_the_sentence_rounded.total_seconds()*frame_rate)
      start_frame_sentence = round(start_time_of_the_sentence_rounded.total_seconds()*frame_rate)
      number_of_frames_inside_sentence = (end_frame_sentence - start_frame_sentence)

      for number_of_word_within_sentence in range(0,words_within_sentence.shape[0]):
          number_of_frames_inside_word=0

          start_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].start_time.total_seconds()*frame_rate)/frame_rate))
          end_time_of_the_word_rounded = timedelta(seconds=(round(words_within_sentence.iloc[number_of_word_within_sentence].end_time.total_seconds()*frame_rate)/frame_rate))
          end_frame_word = round(end_time_of_the_word_rounded.total_seconds()*frame_rate)
          start_frame_word = round(start_time_of_the_word_rounded.total_seconds()*frame_rate)

          number_of_frames_inside_word=(end_frame_word - start_frame_word)


          number_of_frames_inside_word_accumulated = number_of_frames_inside_word_accumulated + number_of_frames_inside_word

      #---------------------------------------------------------------------
      #creating a vector of 0's or 1's depending on the label of every word


          if words_within_sentence.iloc[number_of_word_within_sentence]['label']=='ME':

                temporal_vector_label.extend(np.ones(number_of_frames_inside_word,dtype=int))

          else:
                temporal_vector_label.extend(np.zeros(number_of_frames_inside_word,dtype=int))
          # print("word",number_of_frames_inside_word)
      # print(temporal_vector_label)

      #---------------------------------------------------------------------------------------

      # print("number_of_frames_inside_word_accumulated",number_of_frames_inside_word_accumulated)

      #---------------------------------------------------------------------------------------
      #saving the entire vector of every sentence in a Df

      if number_of_frames_inside_sentence == number_of_frames_inside_word_accumulated : #making sure no frame is missing
        #making sure no frame is missing

          new_row = { 'label': sentence_row.label,
                      'gt_label_per_word': [temporal_vector_label]}

          Dataframe_to_add_data = pd.DataFrame(new_row)

          gt = pd.concat([gt, Dataframe_to_add_data], ignore_index = True)

      else:

        print("the number of total frames is not the same as the ones inside every word")

      #-------------------------------------------------------------------------------------

  print("accumulated number of frames inside a word",number_of_frames_inside_word_accumulated_total)
  print("accumulated number of frames inside a sentence",number_of_frames_inside_sentence_total)

  return gt,DataFrame_of_sentences
