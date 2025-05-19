import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict, get_data_dict_for_dataaug
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time, get_test_loss
from utils import mode_filter,func_eval_for_dataaug
import ast
import shutil
from pathlib import Path
from metrics_sign_seg import metrics_own

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = 1 #len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        for name,param in self.model.named_parameters():
            print(name,param.shape)



    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, checkpoint_path,layers_to_not_load ,layers_to_unfreeze,log_train_results=True):

        


        # if isinstance(self.device, list) and len(self.device) > 1:
        #     # Use DataParallel for multiple devices
        #     self.model = torch.nn.DataParallel(self.model, device_ids=self.device)
        #     # Move model to the first device
        #     self.model.to(f'cuda:{self.device[0]}')
        # else:
        #     # Move model to a single device
        #     self.model.to(self.device)

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()




        restore_epoch = -1
        step = 1
        # if checkpoint_path:
        #     if os.path.exists(checkpoint_path):
        #         saved_state = torch.load(checkpoint_path)
        #         self.model.load_state_dict(saved_state)
        #         if 'optimizer' in saved_state:
        #             optimizer.load_state_dict(saved_state['optimizer'])
        #         if 'epoch' in saved_state:
        #             restore_epoch = saved_state['epoch']
        #         if 'step' in saved_state:
        #             step = saved_state['step']
        #         print(f'Loaded checkpoint from {checkpoint_path}')

        # if checkpoint_path and os.path.exists(checkpoint_path):
            
        #     pretrain_state_dict = torch.load(checkpoint_path, map_location=device)
        #     model_dict = self.model.state_dict()

        #     # Exclude layers specified in layers_to_unfreeze
        #     pretrain_dict = {k: v for k, v in pretrain_state_dict.items() if k not in layers_to_not_load and v.shape == model_dict[k].shape}
        #     model_dict.update(pretrain_dict)
        #     self.model.load_state_dict(model_dict)
        #     print(f'Loaded pre-trained weights excluding layers: {layers_to_not_load}')
        


        for name, param in self.model.named_parameters(): #unfreeze all params
            param.requires_grad = True #used to be false

    # infreeze specified layers
        # if layers_to_unfreeze:
            # for name, param in self.model.named_parameters():
            #     if any(layer in name for layer in layers_to_unfreeze):
            #         param.requires_grad = True
            #         print(f'Layer {name} unfrozen.')

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights,pos_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            pos_weights = torch.tensor(pos_weights)
            # ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
            # ce_criterion= nn.BCEWithLogitsLoss(weight=class_weights, reduction='none') # no funciona
            ce_criterion= nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')


            print("class_weights,pos_weights",class_weights.shape,pos_weights)
        else:
            ce_criterion = nn.BCEWithLogitsLoss(reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')  #nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
            
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
        logger = SummaryWriter(result_dir)

        ########## TRAINING #################
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.model.train()
            
            epoch_running_loss = 0
            epoch_running_loss_f1s = 0
            epoch_running_loss_f1b = 0
            
            for _, data in enumerate(train_train_loader):

                feature, label, boundary, video, gt_eval,sequence_of_events = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)
                
                loss_dict,sign_seg_dict = self.model.get_training_loss(feature, 
                    event_gt=label.unsqueeze(1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label,
                    gt_eval=gt_eval,
                    sequence_of_events=sequence_of_events
                )

                # print("loss_dict",loss_dict)
                # print("loss sign seg",sign_seg_dict)

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T]) 
                # ##################

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v


                

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k} per batch_size', loss_weights[k] * v.item() / batch_size, step) # pondered loss per batch
                    logger.add_scalar('Train-Total per batch_size', total_loss.item() / batch_size, step)

                total_loss /= batch_size # total loss = total_loss/batch/_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                epoch_running_loss_f1b+=sign_seg_dict['mean_f1_b']
                epoch_running_loss_f1s+=sign_seg_dict['mean_f1s']
                
                if step % batch_size == 0: # within how many samples to update the parameters of the model?
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1 # counting the number of samples passing to it so the resulting number will be epoch*len(dataset)


            ##### Calcultaing loss per every epoch
            #   
            epoch_running_loss /= len(train_train_dataset)
            epoch_running_loss_f1s/=len(train_train_dataset)
            epoch_running_loss_f1b/=len(train_train_dataset)

            logger.add_scalar(f'Total Train loss (pondered) per epoch', epoch_running_loss, epoch)
            logger.add_scalar(f'Train_vs_Test/Train F1B per epoch', epoch_running_loss_f1b, epoch)
            logger.add_scalar(f'Train_vs_Test/Train F1S per epoch', epoch_running_loss_f1s, epoch)
            # logger.add_scalar(f'Comparison/Train_vs_Test_F1B/Train_F1B_per_epoch', epoch_running_loss_f1b, epoch)
            # logger.add_scalar(f'Train_vs_Test/Train_F1S', epoch_running_loss_f1s, epoch)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss} - F1s {epoch_running_loss_f1s} - F1b {epoch_running_loss_f1b}')

        ########################### SAVING MODEL ###################################

            if result_dir: #generating dict of the model every epoch

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0: # saving model every given number of epochs

                if result_dir:

                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        

        ################## TEST ################################ original: every given number of epochs the model performs TEST
        
                # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
            for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,logger=logger, epoch=epoch, loss_weights=loss_weights,pos_weight=pos_weights,
                        result_dir=result_dir, model_path=None)
                    
                    # print("HEREEEEEEEEEE")
                    
                    # print("test_result_dict",test_result_dict)

                    if result_dir:
                        for k,v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                    #     np.save(os.path.join(result_dir, 
                    #         f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k,v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


                    # if log_train_results:

                    #     train_result_dict = self.test(
                    #         train_test_dataset, mode, device, label_dir,
                    #         result_dir=result_dir, model_path=None)

                    #     if result_dir:
                    #         for k,v in train_result_dict.items():
                    #             logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                    #         np.save(os.path.join(result_dir, 
                    #             f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                    #     for k,v in train_result_dict.items():
                    #         print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if result_dir:
            logger.close()


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', "None"])


        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature, label, boundary, video,gt_eval,_ = test_dataset[video_idx]

         

            # print("feature",feature)
            # print("label",label)
            # print("video",video)
            # print("gt_eval",gt_eval)

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device)) 
                       for i in range(len(feature))] # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = self.model.ddim_sample(feature[0].to(device), seed) # output is a list of tuples
                output = output.cpu()
                # left_offset = self.sample_rate // 2
                # right_offset = (self.sample_rate - 1) // 2
                ###############
                
                # for i, elem in enumerate(output):
                #     print("Shape of element", i, ":", elem.shape)
                # #############

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [self.model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
                output = [output.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            # assert(output[0].shape[0] == 1)

            # min_len = min([i.shape[2] for i in output])
            # output = [i[:,:,:min_len] for i in output]
            # output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            # output = output.mean(0).numpy()

            #########################
            # print(output, output.shape)
            # output = np.argmax(output, 0)#s
            # print(output)
            ##########################


            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            # output = np.argmax(output, 0)


            ################
            # print(output, output.shape)
            #################

            # output = restore_full_sequence(output, 
            #     full_len=label.shape[-1], 
            #     left_offset=left_offset, 
            #     right_offset=right_offset, 
            #     sample_rate=self.sample_rate
            # )

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()
            output = output.squeeze(0).view(-1)

            # print("output",output,output.shape)



            # print(label.shape)

            assert(output.shape == label.shape)
            
            return video, output, label,gt_eval,boundary


    def test(self, test_dataset, mode, device, label_dir, logger,epoch,loss_weights,pos_weight,result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        # loss_weights_test={'decoder_ce_loss_test': loss_weights['decoder_ce_loss'],
        #                     'decoder_mse_loss_test': loss_weights['decoder_mse_loss'],
        #                       'decoder_boundary_loss_test': loss_weights['decoder_boundary_loss']}
        
        loss_weights_test={
                           'new_decoder_ce_loss_test': loss_weights['decoder_ce_loss'],
                            'decoder_mse_loss_test': loss_weights['decoder_mse_loss'],
                              'decoder_boundary_loss_test': loss_weights['decoder_boundary_loss']}



        sign_seg_dict_test={}
        dict_loss_test={}
        epoch_running_loss=0
        epoch_running_loss_f1s=0
        epoch_running_loss_f1b=0
        total_loss = 0

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                total_loss = 0
                
                video, pred, label,gt_eval,boundary_gt = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)
            

                dict_loss_test=get_test_loss(pred,label, boundary_gt,pos_weight=pos_weight)
               
                # print(dict_loss_test)
                # print(loss_weights)

                for k,v in dict_loss_test.items():
                    total_loss += loss_weights_test[k] * v
                
                epoch_running_loss+= total_loss.item()


                pred=  np.where(np.asarray(pred) > 0.5, 1, 0)

                sign_seg_dict_test=metrics_own(pred,label,gt_eval)

                epoch_running_loss_f1b+=sign_seg_dict_test['mean_f1_b']
                epoch_running_loss_f1s+=sign_seg_dict_test['mean_f1s']

                pred = [self.event_list[int(i)] for i in pred]

                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()
        
            epoch_running_loss /= len(test_dataset)
            epoch_running_loss_f1s/=len(test_dataset)
            epoch_running_loss_f1b/=len(test_dataset)

            logger.add_scalar(f'Total Test loss (pondered) per epoch', epoch_running_loss, epoch)
            logger.add_scalar(f'Train_vs_Test/Test F1B per epoch', epoch_running_loss_f1b, epoch)
            logger.add_scalar(f'Train_vs_Test/Test F1S per epoch', epoch_running_loss_f1s, epoch)


        print(f'Epoch {epoch} -Test Running Loss {epoch_running_loss} - F1s {epoch_running_loss_f1s} - F1b {epoch_running_loss_f1b}')
        # print(f'Epoch {epoch} -Test Running Loss - F1s {epoch_running_loss_f1s} - F1b {epoch_running_loss_f1b}')

        

        acc, f1s = func_eval_for_dataaug(
            label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)
        
        # acc, edit, f1s = func_eval(
        #     label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)

        result_dict = {
            'Acc': acc,
            #'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    # parser.add_argument('--device', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


    split_sentence="split_sentence_0"   
    
    set_name="set_21_onlyreplicate"
    set_name_label="set_21_onlyreplicate_refined_labels" # labels dont change depending on the window
    
    # feature_dir = os.path.join(root_data_dir, dataset_name, 'features',split_sentence,set_name)
    feature_root_data_dir = os.path.join(root_data_dir, dataset_name, 'features',split_sentence)
    # data_aug_directories=["set_21_onlyreplicate","set_21_replicated_rotated_from10_10","set_21_replicated_zoomedin_zoomout_0.8_1.2"]
    # data_aug_directories=["set_21_onlyreplicate","set_21_replicated_rotated_from10_10","set_21_translation_x_y_15_15"]
    data_aug_directories=["set_21_onlyreplicate","set_21_replicated_rotated_from10_10","set_21_replicated_zoomedin_zoomout_0.8_1.2","set_21_translation_x_y_15_15"]




    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth',split_sentence,set_name_label)
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')
    result_dir=os.path.join("/media/iot/ML/Summy/final_version_organized/DiffAct/train_layers_last_layer",dataset_name,'augmented_ver_rotated_zoom_translation_strict_split1_1')
    # checkpoint_path=Path("/media/iot/ML/Summy/final_version_organized/DiffAct/trained_models/50salads-Trained-S1/release.model")
    # checkpoint_path=Path("/media/iot/ML/Summy/final_version_organized/DiffAct/trained_models/50salads-Trained-S2/release.model")
    checkpoint_path=Path("/media/iot/ML/Summy/final_version_organized/DiffAct/trained_models/Breakfast-Trained-S1/release.model")
    # checkpoint_path=Path("/media/iot/ML/Summy/final_version_organized/DiffAct/trained_models/GTEA-Trained-S1/release.model")
    layers_to_unfreeze=[
                          "encoder.conv_in.weight", "encoder.conv_in.bias", 
                        "encoder.encoder.layers.0.conv_block.0.weight", "encoder.encoder.layers.0.conv_block.0.bias",
                        "encoder.encoder.layers.0.att_linear_q.weight","encoder.encoder.layers.0.att_linear_q.bias",
                        "encoder.encoder.layers.0.att_linear_k.weight", "encoder.encoder.layers.0.att_linear_k.bias",
                        "encoder.encoder.layers.0.att_linear_v.weight","encoder.encoder.layers.0.att_linear_v.bias",
                        "encoder.encoder.layers.0.ffn_block.0.weight", "encoder.encoder.layers.0.ffn_block.0.bias",
                        "encoder.encoder.layers.0.ffn_block.2.weight","encoder.encoder.layers.0.ffn_block.2.bias",
                        "encoder.encoder.layers.1.conv_block.0.weight","encoder.encoder.layers.1.conv_block.0.bias",
                        "encoder.encoder.layers.1.att_linear_q.weight","encoder.encoder.layers.1.att_linear_q.bias",
                        "encoder.encoder.layers.1.att_linear_k.weight","encoder.encoder.layers.1.att_linear_k.bias",
                        "encoder.encoder.layers.1.att_linear_v.weight","encoder.encoder.layers.1.att_linear_v.bias",
                        "encoder.encoder.layers.1.ffn_block.0.weight","encoder.encoder.layers.1.ffn_block.0.bias",
                        "encoder.encoder.layers.1.ffn_block.2.weight","encoder.encoder.layers.1.ffn_block.2.bias",
                        "encoder.encoder.layers.2.conv_block.0.weight","encoder.encoder.layers.2.conv_block.0.bias",
                        "encoder.encoder.layers.2.att_linear_q.weight","encoder.encoder.layers.2.att_linear_q.bias",
                        "encoder.encoder.layers.2.att_linear_k.weight","encoder.encoder.layers.2.att_linear_k.bias",
                        "encoder.encoder.layers.2.att_linear_v.weight","encoder.encoder.layers.2.att_linear_v.bias",
                        "encoder.encoder.layers.2.ffn_block.0.weight","encoder.encoder.layers.2.ffn_block.0.bias",
                        "encoder.encoder.layers.2.ffn_block.2.weight","encoder.encoder.layers.2.ffn_block.2.bias",
                        "encoder.encoder.layers.3.conv_block.0.weight","encoder.encoder.layers.3.conv_block.0.bias",
                        "encoder.encoder.layers.3.att_linear_q.weight","encoder.encoder.layers.3.att_linear_q.bias",
                        "encoder.encoder.layers.3.att_linear_k.weight","encoder.encoder.layers.3.att_linear_k.bias",
                        "encoder.encoder.layers.3.att_linear_v.weight","encoder.encoder.layers.3.att_linear_v.bias",
                        "encoder.encoder.layers.3.ffn_block.0.weight","encoder.encoder.layers.3.ffn_block.0.bias",
                        "encoder.encoder.layers.3.ffn_block.2.weight","encoder.encoder.layers.3.ffn_block.2.bias",
                        "encoder.encoder.layers.4.conv_block.0.weight","encoder.encoder.layers.4.conv_block.0.bias",
                        "encoder.encoder.layers.4.att_linear_q.weight","encoder.encoder.layers.4.att_linear_q.bias",
                        "encoder.encoder.layers.4.att_linear_k.weight","encoder.encoder.layers.4.att_linear_k.bias",
                        "encoder.encoder.layers.4.att_linear_v.weight","encoder.encoder.layers.4.att_linear_v.bias",
                        "encoder.encoder.layers.4.ffn_block.0.weight","encoder.encoder.layers.4.ffn_block.0.bias",
                        "encoder.encoder.layers.4.ffn_block.2.weight","encoder.encoder.layers.4.ffn_block.2.bias",
                        "encoder.encoder.layers.5.conv_block.0.weight","encoder.encoder.layers.5.conv_block.0.bias",
                        "encoder.encoder.layers.5.att_linear_q.weight","encoder.encoder.layers.5.att_linear_q.bias",
                        "encoder.encoder.layers.5.att_linear_k.weight","encoder.encoder.layers.5.att_linear_k.bias",
                        "encoder.encoder.layers.5.att_linear_v.weight","encoder.encoder.layers.5.att_linear_v.bias",
                        "encoder.encoder.layers.5.ffn_block.0.weight","encoder.encoder.layers.5.ffn_block.0.bias",
                        "encoder.encoder.layers.5.ffn_block.2.weight","encoder.encoder.layers.5.ffn_block.2.bias",
                        "encoder.encoder.layers.6.conv_block.0.weight","encoder.encoder.layers.6.conv_block.0.bias",
                        "encoder.encoder.layers.6.att_linear_q.weight","encoder.encoder.layers.6.att_linear_q.bias",
                        "encoder.encoder.layers.6.att_linear_k.weight","encoder.encoder.layers.6.att_linear_k.bias",
                        "encoder.encoder.layers.6.att_linear_v.weight","encoder.encoder.layers.6.att_linear_v.bias",
                        "encoder.encoder.layers.6.ffn_block.0.weight","encoder.encoder.layers.6.ffn_block.0.bias",
                        "encoder.encoder.layers.6.ffn_block.2.weight","encoder.encoder.layers.6.ffn_block.2.bias",
                        "encoder.encoder.layers.7.conv_block.0.weight","encoder.encoder.layers.7.conv_block.0.bias",
                        "encoder.encoder.layers.7.att_linear_q.weight","encoder.encoder.layers.7.att_linear_q.bias",
                        "encoder.encoder.layers.7.att_linear_k.weight","encoder.encoder.layers.7.att_linear_k.bias",
                        "encoder.encoder.layers.7.att_linear_v.weight","encoder.encoder.layers.7.att_linear_v.bias",
                        "encoder.encoder.layers.7.ffn_block.0.weight","encoder.encoder.layers.7.ffn_block.0.bias",
                        "encoder.encoder.layers.7.ffn_block.2.weight","encoder.encoder.layers.7.ffn_block.2.bias",
                        "encoder.encoder.layers.8.conv_block.0.weight","encoder.encoder.layers.8.conv_block.0.bias",
                        "encoder.encoder.layers.8.att_linear_q.weight","encoder.encoder.layers.8.att_linear_q.bias",
                        "encoder.encoder.layers.8.att_linear_k.weight","encoder.encoder.layers.8.att_linear_k.bias",
                        "encoder.encoder.layers.8.att_linear_v.weight","encoder.encoder.layers.8.att_linear_v.bias",
                        "encoder.encoder.layers.8.ffn_block.0.weight","encoder.encoder.layers.8.ffn_block.0.bias",
                        "encoder.encoder.layers.8.ffn_block.2.weight","encoder.encoder.layers.8.ffn_block.2.bias",
                        "encoder.encoder.layers.9.conv_block.0.weight","encoder.encoder.layers.9.conv_block.0.bias",
                        "encoder.encoder.layers.9.att_linear_q.weight","encoder.encoder.layers.9.att_linear_q.bias",
                        "encoder.encoder.layers.9.att_linear_k.weight","encoder.encoder.layers.9.att_linear_k.bias",
                        "encoder.encoder.layers.9.att_linear_v.weight","encoder.encoder.layers.9.att_linear_v.bias",
                        "encoder.encoder.layers.9.ffn_block.0.weight","encoder.encoder.layers.9.ffn_block.0.bias",
                        "encoder.encoder.layers.9.ffn_block.2.weight","encoder.encoder.layers.9.ffn_block.2.bias",
                        "encoder.encoder.layers.10.conv_block.0.weight",
"encoder.encoder.layers.10.conv_block.0.bias", 
"encoder.encoder.layers.10.att_linear_q.weight",
"encoder.encoder.layers.10.att_linear_q.bias",
"encoder.encoder.layers.10.att_linear_k.weight",
"encoder.encoder.layers.10.att_linear_k.bias", 
"encoder.encoder.layers.10.att_linear_v.weight", 
"encoder.encoder.layers.10.att_linear_v.bias", 
"encoder.encoder.layers.10.ffn_block.0.weight",
"encoder.encoder.layers.10.ffn_block.0.bias", 
"encoder.encoder.layers.10.ffn_block.2.weight", 
"encoder.encoder.layers.10.ffn_block.2.bias" 
"encoder.encoder.layers.11.conv_block.0.weight", 
"encoder.encoder.layers.11.conv_block.0.bias", 
"encoder.encoder.layers.11.att_linear_q.weight", 
"encoder.encoder.layers.11.att_linear_q.bias", 
"encoder.encoder.layers.11.att_linear_k.weight", 
"encoder.encoder.layers.11.att_linear_k.bias", 
"encoder.encoder.layers.11.att_linear_v.weight", 
"encoder.encoder.layers.11.att_linear_v.bias", 
"encoder.encoder.layers.11.ffn_block.0.weight", 
"encoder.encoder.layers.11.ffn_block.0.bias", 
"encoder.encoder.layers.11.ffn_block.2.weight",
"encoder.encoder.layers.11.ffn_block.2.bias",
                        "encoder.conv_out.weight","encoder.conv_out.bias",
                        "decoder.time_in.0.weight","decoder.time_in.0.bias",
                        "decoder.time_in.1.weight","decoder.time_in.1.bias",
                        "decoder.conv_in.weight","decoder.conv_in.bias",
                        "decoder.module.time_proj.weight","decoder.module.time_proj.bias",
                        "decoder.module.layers.0.conv_block.0.weight","decoder.module.layers.0.att_linear_q.weight",
                        "decoder.module.layers.0.att_linear_q.bias","decoder.module.layers.0.att_linear_k.weight",
                        "decoder.module.layers.0.att_linear_k.bias","decoder.module.layers.0.att_linear_v.weight",
                        "decoder.module.layers.0.att_linear_v.weight","decoder.module.layers.0.att_linear_v.bias",
                        "decoder.module.layers.0.ffn_block.0.weight","decoder.module.layers.0.ffn_block.0.bias",
                        "decoder.module.layers.0.ffn_block.2.weight","decoder.module.layers.0.ffn_block.2.bias",
                        "decoder.module.layers.1.conv_block.0.weight","decoder.module.layers.1.att_linear_q.weight",
                        "decoder.module.layers.1.att_linear_q.bias","decoder.module.layers.1.att_linear_k.weight",
                        "decoder.module.layers.1.att_linear_k.bias","decoder.module.layers.1.att_linear_v.weight",
                        "decoder.module.layers.1.att_linear_v.weight","decoder.module.layers.1.att_linear_v.bias",
                        "decoder.module.layers.1.ffn_block.0.weight","decoder.module.layers.1.ffn_block.0.bias",
                        "decoder.module.layers.1.ffn_block.2.weight","decoder.module.layers.1.ffn_block.2.bias",
                        "decoder.module.layers.2.conv_block.0.weight","decoder.module.layers.2.att_linear_q.weight",
                        "decoder.module.layers.2.att_linear_q.bias","decoder.module.layers.2.att_linear_k.weight",
                        "decoder.module.layers.2.att_linear_k.bias","decoder.module.layers.2.att_linear_v.weight",
                        "decoder.module.layers.2.att_linear_v.weight","decoder.module.layers.2.att_linear_v.bias",
                        "decoder.module.layers.2.ffn_block.0.weight","decoder.module.layers.2.ffn_block.0.bias",
                        "decoder.module.layers.2.ffn_block.2.weight","decoder.module.layers.2.ffn_block.2.bias",
                        "decoder.module.layers.3.conv_block.0.weight","decoder.module.layers.3.att_linear_q.weight",
                        "decoder.module.layers.3.att_linear_q.bias","decoder.module.layers.3.att_linear_k.weight",
                        "decoder.module.layers.3.att_linear_k.bias","decoder.module.layers.3.att_linear_v.weight",
                        "decoder.module.layers.3.att_linear_v.weight","decoder.module.layers.3.att_linear_v.bias",
                        "decoder.module.layers.3.ffn_block.0.weight","decoder.module.layers.3.ffn_block.0.bias",
                        "decoder.module.layers.3.ffn_block.2.weight","decoder.module.layers.3.ffn_block.2.bias",
                        "decoder.module.layers.4.conv_block.0.weight","decoder.module.layers.4.att_linear_q.weight",
                        "decoder.module.layers.4.att_linear_q.bias","decoder.module.layers.4.att_linear_k.weight",
                        "decoder.module.layers.4.att_linear_k.bias","decoder.module.layers.4.att_linear_v.weight",
                        "decoder.module.layers.4.att_linear_v.weight","decoder.module.layers.4.att_linear_v.bias",
                        "decoder.module.layers.4.ffn_block.0.weight","decoder.module.layers.4.ffn_block.0.bias",
                        "decoder.module.layers.4.ffn_block.2.weight","decoder.module.layers.4.ffn_block.2.bias",
                        "decoder.module.layers.5.conv_block.0.weight","decoder.module.layers.5.att_linear_q.weight",
                        "decoder.module.layers.5.att_linear_q.bias","decoder.module.layers.5.att_linear_k.weight",
                        "decoder.module.layers.5.att_linear_k.bias","decoder.module.layers.5.att_linear_v.weight",
                        "decoder.module.layers.5.att_linear_v.weight","decoder.module.layers.5.att_linear_v.bias",
                        "decoder.module.layers.5.ffn_block.0.weight","decoder.module.layers.5.ffn_block.0.bias",
                        "decoder.module.layers.5.ffn_block.2.weight","decoder.module.layers.5.ffn_block.2.bias",
                        "decoder.module.layers.6.conv_block.0.weight","decoder.module.layers.6.att_linear_q.weight",
                        "decoder.module.layers.6.att_linear_q.bias","decoder.module.layers.6.att_linear_k.weight",
                        "decoder.module.layers.6.att_linear_k.bias","decoder.module.layers.6.att_linear_v.weight",
                        "decoder.module.layers.6.att_linear_v.weight","decoder.module.layers.6.att_linear_v.bias",
                        "decoder.module.layers.6.ffn_block.0.weight","decoder.module.layers.6.ffn_block.0.bias",
                        "decoder.module.layers.6.ffn_block.2.weight","decoder.module.layers.6.ffn_block.2.bias",
                        "decoder.module.layers.7.conv_block.0.weight","decoder.module.layers.7.att_linear_q.weight",
                        "decoder.module.layers.7.att_linear_q.bias","decoder.module.layers.7.att_linear_k.weight",
                        "decoder.module.layers.7.att_linear_k.bias","decoder.module.layers.7.att_linear_v.weight",
                        "decoder.module.layers.7.att_linear_v.weight","decoder.module.layers.7.att_linear_v.bias",
                        "decoder.module.layers.7.ffn_block.0.weight","decoder.module.layers.7.ffn_block.0.bias",
                        "decoder.module.layers.7.ffn_block.2.weight","decoder.module.layers.7.ffn_block.2.bias",
                        "decoder.conv_out.weight","decoder.conv_out.bias"]

    layers_to_not_load=['decoder.conv_out.weight', 'decoder.conv_out.bias','decoder.conv_in.weight','encoder.conv_out.bias',
                        "encoder.conv_out.weight","encoder.conv_out.bias","decoder.conv_in.weight ","decoder.conv_in.bias"]

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    print("num_classes",num_classes)

    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'validation.split{split_id}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    train_data_dict = get_data_dict_for_dataaug(root_dir=feature_root_data_dir,
        feature_dir_list=data_aug_directories, 
        label_dir=label_dir, 
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    test_data_dict = get_data_dict_for_dataaug(
        root_dir=feature_root_data_dir,
        feature_dir_list=data_aug_directories, 
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

   
    
    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )   
    print(trainer.device) 
    print("torch",torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir=label_dir, result_dir=result_dir, 
        log_freq=log_freq, checkpoint_path=checkpoint_path,layers_to_not_load=layers_to_not_load , layers_to_unfreeze=layers_to_unfreeze, log_train_results=log_train_results
    )


#python3 main.py --config configs/some_config.json --device gpu_id

#python3 main_s.py --config configs/50salads_conf.json --device 0