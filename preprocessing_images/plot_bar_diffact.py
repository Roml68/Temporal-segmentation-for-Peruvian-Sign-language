import numpy as np
import matplotlib.pyplot as plt

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

# Function to plot the bar graph

def plot_barcode(class_num, gt=None, pred=None, frame_idx=0, total_frames=100):
    if class_num <= 10:
        color_map = plt.cm.tab10
    elif class_num > 20:
        color_map = plt.cm.gist_ncar
    else:
        color_map = plt.cm.tab20

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, interpolation='nearest', vmin=0, vmax=class_num-1)

    fig = plt.figure(figsize=(18, 4))

    # Plot Ground Truth
    if gt is not None:
        ax1 = fig.add_axes([0, 0.45, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth',fontsize=30)
        ax1.imshow(gt.reshape((1, -1)), **barprops)

        if total_frames > 0:
            current_position = (frame_idx / total_frames) * gt.shape[0]
            

    # Plot Prediction
    if pred is not None:
        ax2 = fig.add_axes([0, 0.15, 1, 0.2], **axprops)
        ax2.set_title('Prediction',fontsize=30)
        ax2.imshow(pred.reshape((1, -1)), **barprops)

        if total_frames > 0:
            current_position = (frame_idx / total_frames) * pred.shape[0]
            
    plt.savefig("/home/summy/University/writtint_article/images/26_video_diffact_test.pdf")
    plt.show()
    

# Functions to load ground truth and prediction labels
def labels_to_array(txt_file):
    with open(txt_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return np.array([1 if label == "ME" else 0 for label in labels])

def labels_to_array1(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    content = ' '.join(lines[1:]).strip()
    labels = content.split()
    return np.array([1 if label == "ME" else 0 for label in labels])

# === Example usage ===

video="78"
path_gt = f"/home/summy/Tesis/dataset/manejar_conflictos/labelsrefined/{video}.txt"
path_pred = f"/media/summy/NEW VOLUME/final_results/two_models_manejar_conflictos_final_test/0_results/prediction/{video}.txt"

gt = labels_to_array(path_gt)
pred = labels_to_array1(path_pred)

# Plot only
plot_barcode(class_num=2, gt=gt, pred=pred, frame_idx=10, total_frames=len(gt))
