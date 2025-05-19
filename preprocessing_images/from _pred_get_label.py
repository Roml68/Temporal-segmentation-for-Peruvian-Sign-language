
import pandas as pd

from datetime import timedelta


input_path="/home/summy/Tesis/dataset/test_data/videos/split_sentence_0/video_segments.txt"

def convert_to_Delta_format(Time):

 
  #Converts time in string format to Delta format, so it is easy to manipulate it
  #...
  #Input
  #-----
  #Time: Time that needs to be converted

  #Output
  #-----
  #TimeDeltaFormat: Time converted

  
  deleted=Time.split('0 days')

  Time=deleted[1]

 

  NumberofSectionsOfTheDate=Time.split(':')


  if len(NumberofSectionsOfTheDate) !=3 :
            raise ValueError("Invalid timestamp format. Should be hh:mm:ss.ms")
  else:
            hours,minutes,seconds_milliseconds = Time.split(':')
        
            seconds,milliseconds = seconds_milliseconds.split('.')

  TimeDeltaFormat = timedelta(seconds=int(seconds), milliseconds=int(milliseconds), minutes=int(minutes), hours=int(hours), days=int(0))

  print(TimeDeltaFormat)

  return TimeDeltaFormat


def time_to_seconds(time_str):

    deleted=time_str.split('0 days')
    time_str=deleted[1]
    h, m, s = map(float, time_str.split(":"))

    # print(h, m, s)
    return timedelta(hours=h, minutes=m, seconds=s)



dataframe_of_segments=pd.read_csv(input_path,delimiter="\t")

dataframe_of_segments=pd.DataFrame(dataframe_of_segments)

dataframe_of_segments["start_time"]=dataframe_of_segments["start_time"].apply(lambda x: time_to_seconds(x))

dataframe_of_segments["end_time"]=dataframe_of_segments["end_time"].apply(lambda x: time_to_seconds(x))


# dataframe_of_segments=dataframe_of_segments.iloc[:3]


df_for_ELAN = pd.DataFrame(columns=['start_time', 'end_time', 'label',"end_frame"])  # Initialize DataFrame with columns

frame_rate = 30.0
index_df_ELAN=0

# print(dataframe_of_segments)
for index, row in dataframe_of_segments.iterrows():

    with open(f'/home/summy/Tesis/dataset/test_data/prediction/{index}.txt', 'r') as file:
        # Read and process the text file
        data_array = []
        for line in file:
            elements = line.strip().split()
            data_array.extend(elements)

        data_array = data_array[5:]


    count_frames = 0

    # print(data_array)
    
    
    start_frame = round(row["start_time"].total_seconds() * frame_rate)
    df_for_ELAN.loc[index_df_ELAN, "start_time"] = row["start_time"]


    
    for i in range(len(data_array)-1):  # Iterate over indices, not values

        # print(i)
        if data_array[i] == data_array[i + 1]:
            count_frames += 1
            
            if i==len(data_array)-2:
                # print(count_frames)
                print("lastttt")
                end_frame = start_frame + count_frames+1

                # time_end_frame = end_frame / frame_rate
                # time_end_frame =timedelta(seconds=round(time_end_frame,3))
                # time_end_frame =timedelta(seconds=time_end_frame)


                df_for_ELAN.loc[index_df_ELAN, "end_frame"] = end_frame

                df_for_ELAN.loc[index_df_ELAN, "end_time"] = row["end_time"]
                df_for_ELAN.loc[index_df_ELAN, "label"] = data_array[i]
                count_frames = 0

                index_df_ELAN=index_df_ELAN+1

                df_for_ELAN.loc[index_df_ELAN, "start_time"] = row["end_time"]

                start_frame=end_frame+1

                




        else:
            
            # print(count_frames)
            end_frame = start_frame + count_frames

            time_end_frame = end_frame / frame_rate
            # time_end_frame =timedelta(seconds=round(time_end_frame,3))
            time_end_frame =timedelta(seconds=time_end_frame)


            df_for_ELAN.loc[index_df_ELAN, "end_frame"] = end_frame

            df_for_ELAN.loc[index_df_ELAN, "end_time"] = time_end_frame
            df_for_ELAN.loc[index_df_ELAN, "label"] = data_array[i]
            count_frames = 0

            index_df_ELAN=index_df_ELAN+1
            
            df_for_ELAN.loc[index_df_ELAN, "start_time"] = time_end_frame

            start_frame=end_frame+1

# Print the resulting DataFrame
print(df_for_ELAN)


df = pd.DataFrame(df_for_ELAN)

# # Function to format each row as required
# def format_row(row):
#     start_time = row['start_time']
#     end_time = row['end_time']
#     label = row['label']
    
#     return f"{label} \n {start_time} - {end_time}"

# # Generate lines for each row
# lines = []
# for index, row in df.iterrows():
#     if pd.notna(row['label']):  # Only process rows with valid labels
#         lines.append(format_row(row))

# # Write lines to a text file
# with open('/media/summy/NEW VOLUME/final_results/last_model_test_data_final_model_base_twodatasets_split0/output.txt', 'w') as file:
#     file.write("\n".join(lines))
# print("Text file 'output.txt' created successfully.")

# def format_row(row):
#     start_time = pd.Timedelta(row['start_time']).total_seconds()
#     end_time = pd.Timedelta(row['end_time']).total_seconds()
#     label = row['label']
#     duration = end_time - start_time
    
#     return f"Subtitle-Tier\t\t{row['start_time']}\t{start_time:.3f}\t{row['end_time']}\t{end_time:.3f}\t{duration:.3f}\t{label}"

# # Generate lines for each row
# lines = []
# for index, row in df.iterrows():
#     lines.append(format_row(row))

# # Write lines to a text file
# with open('/media/summy/NEW VOLUME/final_results/last_model_test_data_final_model_base_twodatasets_split0/output.txt', 'w') as file:
#     file.write("\n".join(lines))

# print("Text file 'output.txt' created successfully.")

def format_row(row):
    start_time = pd.Timedelta(row['start_time']).total_seconds()
    end_time = pd.Timedelta(row['end_time']).total_seconds()
    label = row['label']
    duration = end_time - start_time
    
    return f"{start_time:.3f}\t{end_time:.3f}\t{duration:.3f}\t{label}"

# Generate lines for each row
lines = []
for index, row in df.iterrows():
    lines.append(format_row(row))

print(lines)
# Write lines to a text file without headers
with open('/home/summy/Tesis/output.txt', 'w') as file:
    file.write("\n".join(lines))

print("Text file 'output.txt' created successfully.")



