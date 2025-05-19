# import torch
# print(torch.__version__)
# print("cuda: ",torch.cuda.is_available())
# print(torch.version.cuda)

import os
import numpy as np
import torch


def get_gt(paths):
    gt={}
    i=0

    for path in paths:
        tempa=[]
        with open(path, 'r') as fp:
            for line in fp:

                if line.strip() == 'ME':
                    tempa.append(1)
                else:
                    tempa.append(0)
        gt[i]=tempa
        i=i+1
    return gt

def adapt_size_cause_window(gt,window=16):

    gt_processed={}
    for ix in gt:
        gt_processed[ix] = gt[ix][int(window / 2):-int(window / 2) + 1]
    
    return gt_processed


# gt={}
# i=0
# window=16

# for path in paths:
#     tempa=[]
#     number_of_frames=0
#     with open(path, 'r') as fp:
#         for line in fp:

#             if line.strip() == 'ME':
#                 tempa.append(1)
#             else:
#                 tempa.append(0)

#     tempa = tempa[int(window / 2):-int(window / 2) + 1] #cutted to fit the size of features
#     gt[i]=tempa
#     i=i+1

def dilate_boundaries(gt):
  newbound={}
  for gt1 in gt:
        eval_boundaries = []
        gt_temp = [0,0]+gt[gt1]+[0,0]
        con = 0
        for ix in range(2, len(gt[gt1])+2):
            if con:
                con = 0
                continue
            if gt_temp[ix] == 1 and gt_temp[ix+1] == 0 and gt_temp[ix+2] == 0:
                gt_temp[ix+1] = 1
                con = 1
            if gt_temp[ix] == 1 and gt_temp[ix-1] == 0 and gt_temp[ix-2] == 0:
                gt_temp[ix-1] = 1
        #print(len(gt_temp))

        eval_boundaries.extend(gt_temp[2:-2])
        newbound[gt1]=eval_boundaries
    
  return newbound


##### getting metrics

def get_labels_start_end_time1(frame_wise_labels, bg_class):
    """get list of start and end times of each interval/ segment.

    Args:
        frame_wise_labels: list of framewise labels/ predictions.
        bg_class: list of all classes in frame_wise_labels which should be ignored

    Returns:
        labels: list of labels of the segments
        starts: list of start times of the segments
        ends: list of end times of the segments
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    # print("frame_wise_labels[0]",frame_wise_labels)
    if frame_wise_labels[0] != bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] != bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label != bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label != bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def get_boundary_metric(pred, gt, thresholds, bg_class=0):

    p_label, p_start, p_end = get_labels_start_end_time1(pred, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time1(gt, bg_class)

    mean_boundary_width_pred = list(pred).count(1) / len(p_label) if len(p_label) else 0 #ancho del bound pred
    mean_boundary_width_gt = list(gt).count(1) / len(y_label) if len(y_label) else 0

    num_pred = len(p_label)
    num_gt = len(y_label)

    pos_p = [(p_end[i]+p_start[i])/2 for i in range(len(p_label))]
    pos_y = [(y_end[i]+y_start[i])/2 for i in range(len(y_label))]

    # calculate distance matrix
    if len(p_label) > 0:
        dist_all = []
        for p in pos_p:
            dist_all.append([abs(y-p) for y in pos_y])
        dist_arr = np.asarray(dist_all)

        # calculate mean distance
        mean_dist = [np.mean(np.min(dist_arr, 1))]

    else:
        mean_dist = [0]

    # find smallest distances
    dist_choosen = []
    if len(p_label) > 0:
        for ix in range(min(dist_arr.shape[0], dist_arr.shape[1])):

            argmin_row = np.argmin(dist_arr, axis=1)
            min_row = np.min(dist_arr, axis=1)
            min_dist = np.min(min_row)
            argmin_dist = np.argmin(min_row)

            dist_choosen.append(min_dist)

            # delete row and column -> pred-gt pair can't be reused
            dist_arr = np.delete(dist_arr, argmin_dist, 0)
            dist_arr = np.delete(dist_arr, argmin_row[argmin_dist], 1)

    tp_list = []
    fp_list = []
    fn_list = []

    for th in thresholds:
        tp = 0
        fp = 0
        for dist in dist_choosen:
            if dist <= th:
                tp += 1
            else:
                fp += 1

        # more predictions than gt -> count as false positiv
        fp += max(0, len(p_label)-len(dist_choosen))
        # difference between number of true boundaries and correct predicted ones -> false negative
        fn = len(y_label) - tp

        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    return np.asarray(tp_list), np.asarray(fp_list), np.asarray(fn_list), num_pred, num_gt, mean_dist, mean_boundary_width_pred, mean_boundary_width_gt

def get_sign_metric(pred, gt, overlap, bg_class):
    p_label, p_start, p_end = get_labels_start_end_time1(pred, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time1(gt, bg_class)
    iou_all = []

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        iou_all.append((1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))]))

    iou_arr = np.asarray(iou_all)
    iou_choosen = []
    if len(p_label) > 0:
        for ix in range(min(iou_arr.shape[0], iou_arr.shape[1])):
            argmax_row = np.argmax(iou_arr, axis=1)
            max_row = np.max(iou_arr, axis=1)
            max_iou = np.max(max_row)
            argmax_iou = np.argmax(max_row)
            iou_choosen.append(max_iou)
            iou_arr = np.delete(iou_arr, argmax_iou, 0)
            iou_arr = np.delete(iou_arr, argmax_row[argmax_iou], 1)

        diff = max(iou_arr.shape[0], iou_arr.shape[1]) - len(iou_choosen)
    else:
        diff = len(y_label)

    tp_list = []
    fp_list = []
    fn_list = []

    for ol in overlap:
        tp = 0
        fp = 0
        for match in iou_choosen:
            if match > ol:
                tp += 1
            else:
                fp += 1
        fp += max(0, len(p_label)-len(iou_choosen))
        fn = len(y_label) - tp
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    iou_choosen.extend([0]*diff)
    mean_iou = np.mean(iou_choosen)

    return np.asarray(tp_list), np.asarray(fp_list), np.asarray(fn_list), mean_iou


# ########### Applying the functions

# root_directory="/home/summy/Tesis/dataset"
# selected_dataset="manejar_conflictos"

# path_selected_database=os.path.join(root_directory,selected_dataset)


# with open(os.path.join(path_selected_database,"list_of_labels_original.txt"), "r") as labelfile:
#     paths = labelfile.readlines()

# paths = [patha.strip() for patha in paths]
# paths = [os.path.join(path_selected_database,"labels",patha) for patha in paths]


# #################################

# gt_original = get_gt(paths)
# gt = adapt_size_cause_window(gt_original,window=16)

# print(gt_original[170])
# print(gt[170])


# gt_dilatated=dilate_boundaries(gt_original)
# gt_eval=adapt_size_cause_window(gt_dilatated,window=16)

##################################


####################################3

def metrics_own(pred,gt,gt_eval):
        
        assert len(pred)==len(gt) and len(pred)==len(gt_eval)
        
        overlap = list(np.arange(0.4, 0.76, 0.05))

        # print("overlap",overlap)
        thresholds_b = list(range(1, 5))

        tp_list_b, fp_list_b, fn_list_b, num_det, num_gt1, dist, width_pred, width_gt=get_boundary_metric(pred, gt, thresholds_b,0)
        tp_list, fp_list, fn_list, mean_iou=get_sign_metric(pred, gt_eval, overlap, 1)

        f1_sign = []
        # tp_list=tp_list1
        # fp_list=fp_list1
        # fn_list=fn_list1
        for s in range(len(overlap)):

            precision = tp_list[s] / float(tp_list[s]+fp_list[s])
            recall = tp_list[s] / float(tp_list[s]+fn_list[s])
            # print(recall)
            # print(precision)

            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            f1_sign.append(round(f1, 2))

        mean_f1s = np.mean(f1_sign)
        f1_sign = [f1_sign[2],f1_sign[-1]]

        # print("f1_sign",f1_sign)
        # print("mean_f1s",mean_f1s)
        # print("mean_iou",mean_iou)

        f1b = []
        recall_list = []
        precision_list = []
        for s in range(len(thresholds_b)):
            precision_b = tp_list_b[s] / float(tp_list_b[s]+fp_list_b[s])
            recall_b = tp_list_b[s] / float(tp_list_b[s]+fn_list_b[s])
            # print(recall_b)

            f1_b = 2.0 * (precision_b*recall_b) / (precision_b+recall_b)

            # print(s,f1_b)


            f1_b = np.nan_to_num(f1_b)*100
            f1b.append(f1_b)
            recall_list.append(recall_b*100)
            precision_list.append(precision_b*100)

        mean_f1b = round(np.mean(f1b), 2)
        mean_recall_b = round(np.mean(recall_list), 2)
        mean_precision_b = round(np.mean(precision_list), 2)

        mean_dist = np.mean(np.abs(dist))
        mean_width_pred = np.mean(width_pred)
        mean_width_gt = np.mean(width_gt)



        sign_seg_dict ={

            'mean_f1_b':mean_f1b,
            'mean_f1s':mean_f1s,
            'mean_iou':mean_iou,
            'mean_recall_b':mean_recall_b,
            'mean_precision_b':mean_precision_b,
            'mean_dis':mean_dist,
            'mean_width_pred':mean_width_pred,
            'mean_width_gt':mean_width_gt,
            'dist':dist

        }

        return sign_seg_dict
