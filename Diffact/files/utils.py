import os
import json
import random
import torch
import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter
import torch.nn as nn
import pandas as pd
from itertools import product
from tabulate import tabulate


def load_config_file(config_file):

    all_params = json.load(open(config_file))

    if 'result_dir' not in all_params:
        all_params['result_dir'] = 'result'
    
    if 'log_train_results' not in all_params:
        all_params['log_train_results'] = True
    
    if 'soft_label' not in all_params:
        all_params['soft_label'] = None

    if 'postprocess' not in all_params:
        all_params['postprocess'] = {
            'type': None,
            'value': None
        }

    if 'use_instance_norm' not in all_params['encoder_params']:
        all_params['encoder_params']['use_instance_norm'] = False

    if 'detach_decoder' not in all_params['diffusion_params']:
        all_params['diffusion_params']['detach_decoder'] = False

    assert all_params['loss_weights']['encoder_boundary_loss'] == 0

    return all_params


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mode_filter(x, size):
    def modal(P):
        mode = stats.mode(P)
        return mode.mode[0]
    result = generic_filter(x, modal, size)
    return result

############# Modified from ASFormer/MSTCN #################

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def get_labels_start_end_time_for_masking(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i-1)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def create_sequence_with_zeros(length, ranges_to_zero):
    # Create a sequence of ones using PyTorch
    sequence = torch.ones(length, dtype=torch.float32)  # Use float32 to match the original function

    for start, end in ranges_to_zero:
        sequence[:,:,start:end + 1] = 0.0  # Set the specified ranges to zero

    return sequence


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score

 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 
def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
 

def func_eval(label_dir, pred_dir, video_list):
    
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0

    for vid in video_list:
 
        gt_file = os.path.join(label_dir, f'{vid}.txt')
        gt_content = read_file(gt_file).split('\n')[0:-1] # removes the last line cause it is a blank
 
        pred_file = os.path.join(pred_dir, f'{vid}.txt') 
        pred_content = read_file(pred_file).split('\n')[1].split()



 
        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        # edit += edit_score(pred_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    acc = 100 * float(correct) / total
    # edit = (1.0 * edit) / len(video_list)
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
 
    # return acc, edit, f1s
    return acc, f1s

def func_eval_for_plot(label_dir, pred_dir, video_list):
    
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0
    acc_acu={}

    for vid in video_list:
 
        gt_file = os.path.join(label_dir, f'{vid}.txt')
        gt_content = read_file(gt_file).split('\n')[0:-1] # removes the last line cause it is a blank
 
        pred_file = os.path.join(pred_dir, f'{vid}.txt') 
        pred_content = read_file(pred_file).split('\n')[1].split()



 
        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

       

        acc_acu[vid]=100 * float(correct) / total
        # edit += edit_score(pred_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    acc = 100 * float(correct) / total
    # edit = (1.0 * edit) / len(video_list)
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
 
    # return acc, edit, f1s
    return acc, f1s,acc_acu

def func_eval_for_dataaug(label_dir, pred_dir, video_list):
    
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0

    for vid in video_list:

        vid_gt=vid.split('_')[0]
 
        gt_file = os.path.join(label_dir, f'{vid_gt}.txt')
        gt_content = read_file(gt_file).split('\n')[0:-1] # removes the last line cause it is a blank
 
        pred_file = os.path.join(pred_dir, f'{vid}.txt') 
        pred_content = read_file(pred_file).split('\n')[1].split()



 
        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        # edit += edit_score(pred_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    acc = 100 * float(correct) / total
    # edit = (1.0 * edit) / len(video_list)
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
 
    # return acc, edit, f1s
    return acc, f1s


def func_eval_diff_dataset(label_dir, pred_dir, video_list,list_of_datasets,ver_data_aug,root_data_dir):
    
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0

    for vid in video_list:

        # vid_gt=vid.split('_')[0]

        video_name,dataset,_=vid.split('_') #-> data looks like 100_0(dataset)_0(ver_of_data) 

        # ver_of_data=ver_of_data[0]  #eliminando el .txt

        dataset_dir=list_of_datasets[int(dataset)]

 
        gt_file = os.path.join(root_data_dir,dataset_dir,label_dir, '{}.txt'.format(video_name))
        gt_content = read_file(gt_file).split('\n')[0:-1] # removes the last line cause it is a blank
 
        pred_file = os.path.join(pred_dir, f'{vid}.txt') 
        pred_content = read_file(pred_file).split('\n')[1].split()



 
        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        # edit += edit_score(pred_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    acc = 100 * float(correct) / total
    # edit = (1.0 * edit) / len(video_list)
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
 
    # return acc, edit, f1s
    return acc, f1s

def get_test_loss(pred, event_gt, boundary_gt, pos_weight,
          soft_label=None):
        
        decoder_boundary_criterion = nn.BCELoss(reduction='none')
        decoder_ce_criterion = nn.BCELoss(reduction='none')

        decoder_mse_criterion = nn.MSELoss(reduction='none')




        # boundary loss aligment
        #------------------------
        #-------------------------
        # decoder_boundary = 1 - torch.einsum('bicl,bcjl->bijl',  # original
        #     F.softmax(event_out[:,None,:,1:], 2), 
        #     F.softmax(event_out[:,:,None,:-1].detach(), 1)
        # ).squeeze(1)
        #-------------------------
        target = pred[:-1]
        input = pred[:1]

        decoder_boundary = 1 - ((input*target) + ((1-input)*(1-target)))

        # Cross entropy loss
        #-------------------

        if soft_label is None:    # To improve efficiency
            #------------------
            # decoder_ce_loss = decoder_ce_criterion(
            #     event_out.transpose(2, 1).contiguous().view(-1, self.num_classes), 
            #     torch.argmax(event_gt, dim=1).view(-1).long()   # batch_size must = 1
            # )
            #----------------------

            decoder_ce_loss = decoder_ce_criterion(
                pred,  #event_out.transpose(2, 1).contiguous().view(-1, self.num_classes), 
                torch.from_numpy(event_gt) )
            
            new_decoder_loss=[]
            
            for index,gt in enumerate(event_gt):

                if gt==1:

                    new_decoder_loss.append(pos_weight*decoder_ce_loss[index])
                
                else:

                    new_decoder_loss.append(decoder_ce_loss[index])


            
            

            
            
        # else:
        #     soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
        #     for i in range(soft_event_gt.shape[1]):
        #         soft_event_gt[0,i] = gaussian_filter1d(soft_event_gt[0,i], soft_label) # the soft label is not normalized
        #     soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)

        #     decoder_ce_loss = - soft_event_gt * F.log_softmax(event_out, 1)
        #     decoder_ce_loss = decoder_ce_loss.sum(0).sum(0)






        # Temporal smootheness Loss
        # -------------------------
        #--------------------------
        # decoder_mse_loss = torch.clamp(decoder_mse_criterion(
        #     F.log_softmax(event_out[:, :, 1:], dim=1), 
        #     F.log_softmax(event_out.detach()[:, :, :-1], dim=1)), 
        #     min=0, max=16)
        #--------------------------
        decoder_mse_loss = torch.clamp(decoder_mse_criterion(
            pred[1:], 
            pred[:-1]),

            min=0, max=16)

        decoder_boundary_loss = decoder_boundary_criterion(decoder_boundary, boundary_gt[:,:,1:].view(-1).float())
        decoder_boundary_loss = decoder_boundary_loss.mean()

        decoder_ce_loss = decoder_ce_loss.mean()
        decoder_mse_loss = decoder_mse_loss.mean()

        new_decoder_loss=torch.from_numpy(np.array(new_decoder_loss)).mean()


        # loss_dict = {
 
        #     'decoder_ce_loss_test': decoder_ce_loss,
        #     'decoder_mse_loss_test': decoder_mse_loss,
        #     'decoder_boundary_loss_test': decoder_boundary_loss,
        # }

        loss_dict = {
            'new_decoder_ce_loss_test':new_decoder_loss,
            'decoder_mse_loss_test': decoder_mse_loss,
            'decoder_boundary_loss_test': decoder_boundary_loss,
        }

        return loss_dict

############# Visualization #################

def plot_barcode(class_num, gt=None, pred=None, show=True, save_file=None):

    if class_num <= 10:
        color_map = plt.cm.tab10
    elif class_num > 20:
        color_map = plt.cm.gist_ncar
    else:
        color_map = plt.cm.tab20

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, 
                interpolation='nearest', vmin=0, vmax=class_num-1)

    fig = plt.figure(figsize=(18, 4))

    # a horizontal barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.45, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        ax1.imshow(gt.reshape((1, -1)), **barprops)

    if pred is not None:
        ax2 = fig.add_axes([0, 0.15, 1, 0.2], **axprops)
        ax2.set_title('Predicted')
        ax2.imshow(pred.reshape((1, -1)), **barprops)

    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    if show:
        plt.show()

    plt.close(fig)

################

def plot_barcode_plus_curve(class_num, gt=None, pred=None, pred_curve=None, show=True, save_file=None):
    # Set the color map depending on the number of classes
    if class_num <= 10:
        color_map = plt.cm.tab10
    elif class_num > 20:
        color_map = plt.cm.gist_ncar
    else:
        color_map = plt.cm.tab20

    # Set axis and bar properties
    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, 
                    interpolation='nearest', vmin=0, vmax=class_num-1)

    # Create the figure for the barcode plot
    fig = plt.figure(figsize=(18, 4))

    # Plot Ground Truth barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.45, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        ax1.imshow(gt.reshape((1, -1)), **barprops)

    # Plot Predicted barcode and curve
    
    if pred is not None:
            if gt is not None:
                pred_position = 0.15
            else:
                pred_position = 0.3  # Center position when no ground truth is present

            # Plot Predicted barcode
            ax2 = fig.add_axes([0, pred_position, 1, 0.2], **axprops)
            ax2.set_title('Predicted')
            ax2.imshow(pred.reshape((1, -1)), **barprops)

            # Overlay the prediction confidence curve if available
            if pred_curve is not None:
                ax2_curve = ax2.twinx()  # Share the x-axis with the predicted barcode
                ax2_curve.plot(pred_curve, color='black', lw=2)
                ax2_curve.set_ylim(0, 1)  # Ensure the curve is scaled between 0 and 1
                ax2_curve.tick_params(left=False, labelleft=False)


    # Save to file if required
    if save_file is not None:
        fig.savefig(save_file, dpi=400)

    # Show the plot if required
    if show:
        plt.show()

    # Close the plot to prevent memory leakage
    plt.close(fig)



def update_data(existing_data, new_data=None):
    """Updates the dictionary by adding new epoch values."""
    for key, value in new_data.items():
        if key in existing_data:
            existing_data[key].extend(value)
        else:
            existing_data[key] = value  # Create new key if it doesn't exist

def get_best_epoch(df, f1b_col, f1s_col, best_epochs_df,aux_df):
    """Selects the best epoch and appends it to another DataFrame."""
    # Calculate min(F1B, F1S) for each epoch to ensure both are high
    df['min_f1'] = df[[f1b_col, f1s_col]].min(axis=1)

    # Filter for epochs with the highest min(F1B, F1S)
    max_min_f1 = df['min_f1'].max()
    candidates = df[df['min_f1'] == max_min_f1]

    # If there's a tie, select the epoch with the highest max(F1B, F1S)
    if len(candidates) > 1:
        candidates['max_f1'] = candidates[[f1b_col, f1s_col]].max(axis=1)
        best_row = candidates.loc[candidates['max_f1'].idxmax()]
    else:
        best_row = candidates.iloc[0]

    # print("\nBest epoch with the highest balanced F1B and F1S:")
    # print(best_row)

    # Append the best row to the best_epochs_df
    aux_df.update(best_row)

    # print(aux_df)

    aux_df=pd.DataFrame([aux_df])

    if best_epochs_df is not None:
        best_epochs_df = pd.concat([best_epochs_df,aux_df ], ignore_index=True)
    
    print(tabulate(best_epochs_df, headers='keys', tablefmt='pipe'))
    
    return best_epochs_df
    
    


    

    