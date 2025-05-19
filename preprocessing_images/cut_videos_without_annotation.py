
import pandas as pd
import random
from datetime import timedelta, datetime

def seconds_to_time(seconds):
    return str(timedelta(seconds=seconds))

def seconds_to_time1(seconds):
    return timedelta(seconds=seconds)


# video information
start_time_video="00:00:31.600"
end_time_video="00:25:38.307"
frame_rate=30

silencios=[["00:02:24.787","00:04:16.652"],["00:06:38.627","00:06:47.987"],["00:06:58.413","00:07:08.440"],["00:15:14.933","00:15:24.440"]]


min_seconds=seconds_to_time1(6)
max_seconds=seconds_to_time1(10)




# Convert time strings to datetime objects
def time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    milliseconds=time_str.split(".")[1]
    time=int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return timedelta(seconds=(round(time*frame_rate)/frame_rate))








start_time_sec = time_to_seconds(start_time_video)
end_time_sec = time_to_seconds(end_time_video)
silencios_sec = [(time_to_seconds(start), time_to_seconds(end)) for start, end in silencios]

# Generate video segments while avoiding silence ranges
segments = []
current_time = start_time_sec




video_number=0
while current_time < end_time_sec:
    duration = random.uniform(6, 10)
    duration=seconds_to_time1(duration)
    end_segment = current_time + duration

    # Adjust the segment for silence ranges
    for silence_start, silence_end in silencios_sec:
        if current_time < silence_start < end_segment:  # Ends at silence start
            end_segment = silence_start
        elif silence_start <= current_time < silence_end:  # Starts after silence
            current_time = silence_end
            end_segment = current_time + duration
    
    # Ensure the fragment doesn't exceed video bounds
    if end_segment > end_time_sec:
        end_segment = end_time_sec

    # Add segment and handle small fragments
    if segments and end_segment - current_time < min_seconds:
        prev_start, prev_end, name = segments.pop()
        segments.append((prev_start, end_segment, f"{video_number}.mp4"))
    elif end_segment - current_time >= min_seconds:
        segments.append((current_time, end_segment, f"{video_number}.mp4"))
    
    current_time = end_segment

    video_number+=1

# Convert segments to a readable format
formatted_segments = [(str(start), str(end), name) for start, end, name in segments]

# Result DataFrame
import pandas as pd
df = pd.DataFrame(formatted_segments, columns=["start_time", "end_time", "name_of_video"])
print(df[90:100])