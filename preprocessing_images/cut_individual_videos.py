import subprocess
import cv2
import os

def crop_and_rescale_video(input_file_path, output_file_path):
    """
    Cuts the video so just the most important section is considered and also reescalates
    """
    # Crop area (pixels where the signer is located)
    y1, y2 = 170, 1170
    x1, x2 = 320, 1580

    # Construct the ffmpeg command to crop the video
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_file_path,
        "-filter_complex", f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1},scale=220:220",
        "-r", str(29.97002997002997),  # Frame rate
        output_file_path
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True)
        print("FFmpeg command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def generate_video_list(folder_path, output_path):
    """
    Generate a list of all video files (.mp4) in the folder and save it to a text file.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the videos.
    output_txt_path : str
        Path to save the list of video file names.

    """
    output_txt_path=os.path.join(output_path,"list_of_videos_original.txt")

    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    
    with open(output_txt_path, 'w') as file:
        for video in video_files:
            file.write(f"{video}\n")
    
    print(f"List of videos saved to {output_txt_path}")

    return output_txt_path

def process_videos_from_list(folder_path, output_folder, video_list_path):
    """
    Applies the crop and rescale function to videos listed in the given text file.
    Renames the output videos sequentially (0.mp4, 1.mp4, 2.mp4, ...).

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the videos.
    output_folder : str
        Path to the folder where the processed videos will be saved.
    video_list_path : str
        Path to the text file containing the list of video files to process.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the video list from the text file
    with open(video_list_path, 'r') as file:
        video_files = [line.strip() for line in file.readlines()]

    # Process each video and rename the output sequentially
    for idx, video_file in enumerate(video_files):
        input_file_path = os.path.join(folder_path, video_file)
        output_file_path = os.path.join(output_folder, f"{idx}.mp4")  # Sequential naming

        # Apply the crop and rescale function
        crop_and_rescale_video(input_file_path, output_file_path)
    

# # Paths to input and output files
# input_file_path = "/home/summy/Tesis/Segundo_avance_(corregido)/ABRIR/ABRIR_ORACION_1.mp4"
# cropped_output_path = "/home/summy/Tesis/Segundo_avance_(corregido)/ABRIR/cropped_video.mp4"
# rescaled_output_path = "/home/summy/Tesis/Segundo_avance_(corregido)/ABRIR/rescaled_video.mp4"

# # First, crop the video
# get_segment_ver2(input_file_path, cropped_output_path)


root_path="/home/summy/Tesis/dataset/305_PUCP"
folder="raw_data"

videos_folder=os.path.join(root_path,folder)
output_folder=os.path.join(root_path,"videos")

output_path_txt=generate_video_list(videos_folder,root_path)

process_videos_from_list(videos_folder, output_folder, output_path_txt)

