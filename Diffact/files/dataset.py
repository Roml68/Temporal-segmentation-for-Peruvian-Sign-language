import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils import get_labels_start_end_time,get_labels_start_end_time_for_masking
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt#s

# import torch
# torch.cuda.empty_cache()

def get_data_dict(feature_dir, label_dir, video_list, event_list, sample_rate=4, temporal_aug=False, boundary_smooth=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        'gt_eval': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file = os.path.join(feature_dir, '{}.npy'.format(video))
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)

        feature = np.load(feature_file, allow_pickle=True)

        # print("original_features",feature)
        # print("features_size",feature.shape)
        
        if len(feature.shape) == 3:

            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2:
            feature = np.swapaxes(feature, 0, 1)
            feature = np.expand_dims(feature, 0)
            # print(feature.shape)
            
        else:
            raise Exception('Invalid Feature.')
                    
        assert(feature.shape[1] == event_seq_raw.shape[0])
        assert(feature.shape[1] == boundary_seq_raw.shape[0])

        # print("swaped_features_size",feature.shape)
                                
        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext = [
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
                        
        # else:
            # feature = [feature[:,::sample_rate,:]]  
            # event_seq_ext = [event_seq_raw[::sample_rate]]
            # boundary_seq_ext = [boundary_seq_raw[::sample_rate]]

            ##### deleting sampling

            ##################
            # feature=feature
            # event_seq_ext=event_seq_raw
            # boundary_seq_ext=boundary_seq_raw
            ######################




        # ################# get the shape of every element
        # num_elements = len(feature)
        # print("Number of elements in 'feature' list:", num_elements)

        # # Print the shape of each element in the list
        # for i, elem in enumerate(feature):
        #     print("Shape of element", i, ":", elem.shape)

        # ########################################

        data_dict[video]['feature'] = [torch.from_numpy(feature).float()]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()

        ### adding gt eval 

        gt_eval=dilate_boundaries(event_seq_raw)

        data_dict[video]['gt_eval']=torch.from_numpy(gt_eval[0]).float()

        ### adding sequence of event

        sequence_of_events=get_labels_start_end_time_for_masking(event_seq_raw)
    
        data_dict[video]['sequence_of_events']=np.array(sequence_of_events).reshape((3,-1))


        

    return data_dict




def get_data_dict_for_dataaug(root_dir,feature_dir_list, label_dir, video_list, event_list, sample_rate=4, temporal_aug=False, boundary_smooth=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        'gt_eval': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):

        ver_of_data=video.split('_')[1][0]

        video_name=video.split('_')[0]

        feature_dir=feature_dir_list[int(ver_of_data)]
        
        feature_file = os.path.join(root_dir,feature_dir, '{}.npy'.format(video_name))
        event_file = os.path.join(label_dir, '{}.txt'.format(video_name))

        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)

        feature = np.load(feature_file, allow_pickle=True)

        # print("original_features",feature)
        # print("features_size",feature.shape)
        
        if len(feature.shape) == 3:

            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2:
            feature = np.swapaxes(feature, 0, 1)
            feature = np.expand_dims(feature, 0)
            # print(feature.shape)
            
        else:
            raise Exception('Invalid Feature.')
                    
        assert(feature.shape[1] == event_seq_raw.shape[0])
        assert(feature.shape[1] == boundary_seq_raw.shape[0])

        # print("swaped_features_size",feature.shape)
                                
        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext = [
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
                        
        # else:
            # feature = [feature[:,::sample_rate,:]]  
            # event_seq_ext = [event_seq_raw[::sample_rate]]
            # boundary_seq_ext = [boundary_seq_raw[::sample_rate]]

            ##### deleting sampling

            ##################
            # feature=feature
            # event_seq_ext=event_seq_raw
            # boundary_seq_ext=boundary_seq_raw
            ######################




        # ################# get the shape of every element
        # num_elements = len(feature)
        # print("Number of elements in 'feature' list:", num_elements)

        # # Print the shape of each element in the list
        # for i, elem in enumerate(feature):
        #     print("Shape of element", i, ":", elem.shape)

        # ########################################

        data_dict[video]['feature'] = [torch.from_numpy(feature).float()]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()

        ### adding gt eval 

        gt_eval=dilate_boundaries(event_seq_raw)

        data_dict[video]['gt_eval']=torch.from_numpy(gt_eval[0]).float()

        ### adding sequence of event

        sequence_of_events=get_labels_start_end_time_for_masking(event_seq_raw)
    
        data_dict[video]['sequence_of_events']=np.array(sequence_of_events).reshape((3,-1))


        

    return data_dict

def get_data_dict_for_diff_dataset(root_dir,dataset_list,feature_dir_list,feature_dir, label_dir, video_list, event_list, sample_rate=4, temporal_aug=False, boundary_smooth=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        'gt_eval': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):

        # ver_of_data=video.split('_')[2][0]

        # dataset=video.split('_')[1]

        # video_name=video.split('_')[0]

     
        video_name,dataset,ver_of_data=video.split('_') #-> data looks like 100_0(dataset)_0(ver_of_data) 

        ver_of_data=ver_of_data[0]  #eliminando el .txt

        feature_ver=feature_dir_list[int(ver_of_data)]

        dataset_dir=dataset_list[int(dataset)]

  



        
        feature_file = os.path.join(root_dir,dataset_dir,feature_dir,feature_ver, '{}.npy'.format(video_name))
        event_file = os.path.join(root_dir,dataset_dir,label_dir, '{}.txt'.format(video_name))


        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)

        feature = np.load(feature_file, allow_pickle=True)

        # print("original_features",feature)
        # print("features_size",feature.shape)
        
        if len(feature.shape) == 3:

            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2:
            feature = np.swapaxes(feature, 0, 1)
            feature = np.expand_dims(feature, 0)
            # print(feature.shape)
            
        else:
            raise Exception('Invalid Feature.')
                    
        assert(feature.shape[1] == event_seq_raw.shape[0])
        assert(feature.shape[1] == boundary_seq_raw.shape[0])

        # print("swaped_features_size",feature.shape)
                                
        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext = [
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
                        
        # else:
            # feature = [feature[:,::sample_rate,:]]  
            # event_seq_ext = [event_seq_raw[::sample_rate]]
            # boundary_seq_ext = [boundary_seq_raw[::sample_rate]]

            ##### deleting sampling

            ##################
            # feature=feature
            # event_seq_ext=event_seq_raw
            # boundary_seq_ext=boundary_seq_raw
            ######################




        # ################# get the shape of every element
        # num_elements = len(feature)
        # print("Number of elements in 'feature' list:", num_elements)

        # # Print the shape of each element in the list
        # for i, elem in enumerate(feature):
        #     print("Shape of element", i, ":", elem.shape)

        # ########################################

        data_dict[video]['feature'] = [torch.from_numpy(feature).float()]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()

        ### adding gt eval 

        gt_eval=dilate_boundaries(event_seq_raw)

        data_dict[video]['gt_eval']=torch.from_numpy(gt_eval[0]).float()

        ### adding sequence of event

        sequence_of_events=get_labels_start_end_time_for_masking(event_seq_raw)
    
        data_dict[video]['sequence_of_events']=np.array(sequence_of_events).reshape((3,-1))


        

    return data_dict

def get_data_test(feature_dir, video_list, event_list, sample_rate=4, temporal_aug=False, boundary_smooth=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file = os.path.join(feature_dir, '{}.npy'.format(video))

        feature = np.load(feature_file, allow_pickle=True)

        # print("original_features",feature)
        # print("features_size",feature.shape)
        
        if len(feature.shape) == 3:

            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2:
            feature = np.swapaxes(feature, 0, 1)
            feature = np.expand_dims(feature, 0)
            # print(feature.shape)
            
        else:
            raise Exception('Invalid Feature.')
                    


        # print("swaped_features_size",feature.shape)
                                
        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
                        

        # ########################################

        data_dict[video]['feature'] = [torch.from_numpy(feature).float()]

        

    return data_dict


def get_boundary_seq(event_seq, boundary_smooth=None):

    boundary_seq = np.zeros_like(event_seq)

    _, start_times, end_times = get_labels_start_end_time([str(int(i)) for i in event_seq])
    boundaries = start_times[1:]
    assert min(boundaries) > 0
    boundary_seq[boundaries] = 1
    boundary_seq[[i-1 for i in boundaries]] = 1
  ############
    # count_number_boundaries=0
    # indices = range(len(boundary_seq))
    # count_number_boundaries=sum(x != 0 for x in boundary_seq)
    
    # print("boundary_seg",boundary_seq[550:750]) #s
    
    # plt.plot(indices,boundary_seq)
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('Index vs. Value in boundary_seq')
    # plt.show()

    ########
    # count=0
    # affected_frames=0
    ##########

    if boundary_smooth !="None":
        boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)

        #################
        # count=sum(x != 0 for x in boundary_seq)
        # print("total nframes part of boundary: ",count)
        # print("number of boundaries",count_number_boundaries/2)
        # affected_frames=count/(count_number_boundaries/2)


        # print("gaussian_filtered_boundary ",boundary_seq) #s
        # print("size of gaussian_filtered_boundary",boundary_seq.size)
        # nonzero_indices = np.nonzero(boundary_seq)[0]

        # plt.scatter(nonzero_indices, boundary_seq[nonzero_indices], marker='o', linestyle='None', color='red')
        # plt.plot(indices,boundary_seq)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Index vs. Value in boundary_seq after gaussian filter of {}'.format(affected_frames))
        # plt.show()
        

        ####################

        # Normalize. This is ugly.
        temp_seq = np.zeros_like(boundary_seq)
        temp_seq[temp_seq.shape[0] // 2] = 1
        temp_seq[temp_seq.shape[0] // 2 - 1] = 1

        
        norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
        boundary_seq[boundary_seq > norm_z] = norm_z
        boundary_seq /= boundary_seq.max()

        
        # print("boundary_normalized_final",boundary_seq)
        # plt.plot(indices,boundary_seq)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Index vs. Value in boundary_seq after normalization')
        # plt.show()

    return boundary_seq

######## got it from MSTCN -> got it from Renz
def dilate_boundaries(gt):
    eval_boundaries = []
    item = gt.tolist()
    gt_temp = [0,0]+item+[0,0]
    con = 0
    for ix in range(2, len(item)+2):
            if con:
                con = 0 
                continue
            if gt_temp[ix] == 1 and gt_temp[ix+1] == 0 and gt_temp[ix+2] == 0:
                gt_temp[ix+1] = 1
                con = 1
            if gt_temp[ix] == 1 and gt_temp[ix-1] == 0 and gt_temp[ix-2] == 0:
                gt_temp[ix-1] = 1
    eval_boundaries.append(gt_temp[2:-2])
    return np.array(eval_boundaries)

#########################

def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
        
    # frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    frame_ticks = np.arange(left_offset, full_len-right_offset)

    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)

    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert(len(frame_ticks) == len(x)) # Rethink this
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out




class VideoFeatureDataset(Dataset):
    def __init__(self, data_dict, class_num, mode):
        super(VideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test','other'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]
    def get_class_weights(self):
        
        full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw'] for v in self.video_list])
        class_counts = np.zeros((2,)) #class_counts = np.zeros((self.class_num,)) 
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
                    
        # class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)
        class_weights = class_counts.sum() / (class_counts  * 2)

        pos_weight = class_counts[0]/class_counts[1]

        # print("class_counts[0]",class_counts[0])
        # print("class_counts[1]",class_counts[1])


        return class_weights, pos_weight

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        if self.mode == 'train':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_raw'] # s
            boundary = self.data_dict[video]['boundary_seq_raw'] # s
            gt_eval=self.data_dict[video]['gt_eval']
            sequence_of_events=self.data_dict[video]['sequence_of_events']


            feature = feature[0]
            feature = feature[0]

            feature = feature.T 
        
            boundary = boundary.unsqueeze(0)
            boundary /= boundary.max()  # normalize again
            
        if self.mode == 'test': # test aún no corregido el tamaño 

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_raw']
            boundary = self.data_dict[video]['boundary_seq_raw']  # boundary_seq_raw not used
            gt_eval=self.data_dict[video]['gt_eval']
            sequence_of_events=self.data_dict[video]['sequence_of_events']

            

            # feature=feature[0].unsqueeze(0)

            # print(feature[0].shape)

            # feature = feature[0]
            # feature = feature[0]    #feature = [torch.swapaxes(i, 1, 2) for i in feature] # [10 x F x T]
            # feature = feature.T 
            feature = [torch.swapaxes(i, 1, 2) for i in feature]
            #por el hecho de que no hay copias hay modificación de tamaño para varias capas, verificar ello 
            # print("feature_later_size", feature.shape)
            label = label.unsqueeze(0)   # 1 X T 
            boundary = boundary.unsqueeze(0).unsqueeze(0)  # [1 x 1 x T]  

        # print("final_shape_feature",  feature.size())
        else:
            feature = self.data_dict[video]['feature']
            feature = [torch.swapaxes(i, 1, 2) for i in feature]

            return feature



        return feature, label, boundary, video,gt_eval,sequence_of_events

    
