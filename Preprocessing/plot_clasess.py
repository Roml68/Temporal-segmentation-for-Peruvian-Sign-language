"""
file that plots a graph comparing the number of frames per classes -- sign and transition -- 
"""
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
SMALL_SIZE = 16
MEDIUM_SIZE = 19
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


directory="/home/summy/Tesis/dataset/manejar_conflictos/labelsrefined"


# Initialize counters
sign_count = 0
me_count = 0

# Pattern to check if a filename contains a number
pattern = re.compile(r'\d')

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .txt file and contains a number in its name
    if filename.endswith('.txt') and pattern.search(filename):
        # Open the file and count lines with "sign" and "ME"
        with open(os.path.join(directory, filename), 'r') as file:
            for line in file:
                # Check if "sign" or "ME" is in the line
                if 'sign' in line:
                    sign_count += 1
                if 'ME' in line:
                    me_count += 1

# Display the counts
print(f"Total lines containing 'sign': {sign_count}")
print(f"Total lines containing 'ME': {me_count}")

print(sign_count/me_count)

labels = ['Seña', 'Transición']
counts = [sign_count, me_count]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, counts, color=['#82d8ff', '#b1b4ff']) 
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
# plt.title('Frecuencia de las clases \"Seña \" y \"Transición\" en el Video \"Manejar Conflictos\"')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.show()
