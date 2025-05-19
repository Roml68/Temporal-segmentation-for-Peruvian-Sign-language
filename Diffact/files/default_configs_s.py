import os
import json
import copy
import argparse
import sys

params_gtea = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"gtea",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False, 
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":  ['full', 'zero', 'boundary03-', 'segment=1', 'segment=1'],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":1e-6,
   "num_epochs":10001,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":1,
   "soft_label": 1.4,
   "log_train_results":False,
   "postprocess":{
      "type":"purge",
      "value":3
   },
}

params_50salads = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"50salads",
   "split_id":1,
   "sample_rate":8,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":7,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":1.0,
      "cond_types":[
         "full",
         "zero",
         "boundary05-",
         "segment=2",
         "segment=2"
      ],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":0,
   "num_epochs":5001,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":20,
   "soft_label": None,
   "log_train_results":False,
   "postprocess":{
      "type":"median", # W
      "value":30 # W
   },
}

params_breakfast = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"breakfast",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":12,
      "num_f_maps":256,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.1,
      "temporal_dropout_rate":0.1,
      "feature_layer_indices":[
         7,
         8,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":128,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":[
         "full",
         "zero",
         "boundary03-",
         "segment=1",
         "segment=1"
      ],
      "detach_decoder":False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.025,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.025,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0001,
   "weight_decay":0,
   "num_epochs":1001,
   "log_freq":20,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":3,
   "soft_label":4,
   "log_train_results":False,
   "postprocess":{
      "type":"median",
      "value":15
   },
}



parser = argparse.ArgumentParser(description='Process split_id from the terminal.')
parser.add_argument('--base_model', type=str)
parser.add_argument('--root_data_dir',type=str)
parser.add_argument('--split_id', type=int)
parser.add_argument('--sample_rate',type=int)
parser.add_argument('--temporal_aug',action='store_true')
parser.add_argument('--encoder_use_instance_norm',action='store_true')
parser.add_argument('--encoder_num_layers',type=int)
parser.add_argument('--encoder_num_f_maps',type=int)
parser.add_argument('--encoder_input_dim',type=int)
parser.add_argument('--encoder_kernel_size',type=int)
parser.add_argument('--encoder_normal_dropout_rate',type=float)
parser.add_argument('--encoder_channel_dropout_rate',type=float)
parser.add_argument('--encoder_temporal_dropout_rate',type=float)
parser.add_argument('--encoder_feature_layer_indices',type=int,nargs='+')
parser.add_argument('--decoder_num_layers',type=int)
parser.add_argument('--decoder_num_f_maps',type=int)
parser.add_argument('--decoder_time_emb_dim',type=int)
parser.add_argument('--decoder_kernel_size',type=int)
parser.add_argument('--decoder_dropout_rate',type=float)
parser.add_argument('--diffusion_timesteps',type=int)
parser.add_argument('--diffusion_sampling_timesteps',type=int)
parser.add_argument('--diffusion_ddim_sampling_eta',type=float)
parser.add_argument('--diffusion_snr_scale',type=float)
parser.add_argument('--diffusion_cond_types',type=str,nargs='+')
parser.add_argument('--detach_decoder',action='store_true')
parser.add_argument('--encoder_ce_loss',type=float)
parser.add_argument('--encoder_mse_loss',type=float)
parser.add_argument('--encoder_boundary_loss',type=float)
parser.add_argument('--decoder_ce_loss',type=float)
parser.add_argument('--decoder_mse_loss',type=float)
parser.add_argument('--decoder_boundary_loss',type=float)
parser.add_argument('--batch_size',type=str)
parser.add_argument('--learning_rate',type=float)
parser.add_argument('--weight_decay',type=int)
parser.add_argument('--num_epochs',type=int)
parser.add_argument('--log_freq',type=int)
parser.add_argument('--class_weighting',action='store_true')
parser.add_argument('--set_sampling_seed',action='store_true')
parser.add_argument('--boundary_smooth',type=float)
parser.add_argument('--soft_label',action='store_true')
parser.add_argument('--log_train_results',action='store_true')
parser.add_argument('--postprocess_type',type=str)
parser.add_argument('--postprocess_value',type=int)





# parser.add_argument('split_id', type=int, nargs='?', help='The split ID to be used. If not provided, will use the default from params_gtea.', default=None)

args = parser.parse_args()

# print(args)

# define which pretrained model to use 

if args.base_model=='gtea':
    params = copy.deepcopy(params_gtea)
    params['naming'] = f'GTEA_conf'
    
elif args.base_model=='50salads':
    params = copy.deepcopy(params_50salads)
    params['naming'] = f'50salads_conf'
elif args.base_model=='breakfast':
    params = copy.deepcopy(params_breakfast)
    params['naming'] = f'breakfast_conf'
else:
    print('error no name of the base model was provided in --base_model')
    parser.print_help()
    sys.exit(1)

params['dataset_name']='manejar_conflictos' #name of the used dataset

# basic configuration

if args.root_data_dir!=None:
        params['root_data_dir']=args.root_data_dir  #root directory
if args.split_id!=None:
        params['split_id'] = args.split_id          #split_id
if args.sample_rate!=None:
        params['sample_rate'] = args.sample_rate    #sample_rate
if args.temporal_aug!=None:
    params['temporal_aug'] = args.temporal_aug      #temporal_augmentation


# encoder parameters

if args.encoder_use_instance_norm!=None:
    params['encoder_params']['use_instance_norm']=args.encoder_use_instance_norm        # use_instance_norm 
if args.encoder_num_layers!=None:
    params['encoder_params']['num_layers']=args.encoder_num_layers                      # encoder number of layers 
if args.encoder_num_f_maps!=None:
    params['encoder_params']['num_f_maps']=args.encoder_num_f_maps                      # encoder number of feature maps
if args.encoder_input_dim!=None:
    params['encoder_params']['input_dim']=args.encoder_input_dim                        # encoder input dimension
if args.encoder_kernel_size!=None:
    params['encoder_params']['kernel_size']=args.encoder_kernel_size                    # encoder kernel size
if args.encoder_normal_dropout_rate!=None:
    params['encoder_params']['normal_dropout_rate']=args.encoder_normal_dropout_rate    # encoder normal dropout rate
if args.encoder_channel_dropout_rate!=None:
    params['encoder_params']['channel_dropout_rate']=args.encoder_channel_dropout_rate  # encoder channel dropout rate
if args.encoder_temporal_dropout_rate!=None:
    params['encoder_params']['temporal_dropout_rate']=args.encoder_temporal_dropout_rate# encoder temporal dropout rate
if args.encoder_feature_layer_indices!=None:
    params['encoder_params']['feature_layer_indices']=args.encoder_feature_layer_indices# encoder feature layer indices --> vector

# decoder parameters

if args.decoder_num_layers!=None:
    params['decoder_params']['num_layers']=args.decoder_num_layers                      # decoder number of layers
if args.decoder_num_f_maps!=None:
    params['decoder_params']['num_f_maps']=args.decoder_num_f_maps                      # decoder number of feature maps
if args.decoder_time_emb_dim!=None:
    params['decoder_params']['time_emb_dim']=args.decoder_time_emb_dim                  # decoder time embedding dimension
if args.decoder_kernel_size!=None:
    params['decoder_params']['kernel_size']=args.decoder_kernel_size                    # decoder kernel size
if args.decoder_dropout_rate!=None:
    params['decoder_params']['dropout_rate']=args.decoder_dropout_rate                  # decoder dropout rate
 
# diffussion parameters 

if args.diffusion_timesteps!=None:
    params['diffusion_params']['timesteps']=args.diffusion_timesteps                    # diffussion timesteps
if args.diffusion_sampling_timesteps!=None:
    params['diffusion_params']['sampling_timesteps']=args.diffusion_sampling_timesteps  # diffusion sampling steps
if args.diffusion_ddim_sampling_eta!=None:
    params['diffusion_params']['ddim_sampling_eta']=args.diffusion_ddim_sampling_eta    # diffusion ddim sampling eta
if args.diffusion_snr_scale!=None:
    params['diffusion_params']['snr_scale']=args.diffusion_snr_scale                    # difussion snr scale
if args.diffusion_cond_types!=None:
    params['diffusion_params']['cond_types']=args.diffusion_cond_types                  # masking condition types

if args.detach_decoder!=None:
    params['detach_decoder']=args.detach_decoder                                        # detach decoder?

# loss_weights parameters

if args.encoder_ce_loss!=None:
    params['loss_weights']['encoder_ce_loss']=args.encoder_ce_loss                      # encoder ce loss
if args.encoder_mse_loss!=None:
    params['loss_weights']['encoder_mse_loss']=args.encoder_mse_loss                    # encoder mse loss 
if args.encoder_boundary_loss!=None:
    params['loss_weights']['encoder_boundary_loss']=args.encoder_boundary_loss          # encoder boundary loss
if args.decoder_ce_loss!=None:
    params['loss_weights']['decoder_ce_loss']=args.decoder_ce_loss                      # decoder ce loss
if args.decoder_mse_loss!=None:
    params['loss_weights']['decoder_mse_loss']=args.decoder_mse_loss                    # decoder mse loss
if args.decoder_boundary_loss!=None:
    params['loss_weights']['decoder_boundary_loss']=args.decoder_boundary_loss          # decoder bounary loss

# about learning of the model

if args.batch_size!=None:
    params['batch_size']=args.batch_size                                                # batch size
if args.learning_rate!=None:
    params['learning_rate']=args.learning_rate                                          # learning rate
if args.weight_decay!=None:
    params['weight_decay']=args.weight_decay                                            # weight decay
if args.num_epochs!=None:
    params['num_epochs']=args.num_epochs                                                # num epochs
if args.log_freq!=None:
    params['log_freq']=args.log_freq                                                    # log freq --> how often save logs
if args.class_weighting!=None:
    params['class_weighting']=args.class_weighting                                      # class weighting 
if args.set_sampling_seed!=None:
    params['set_sampling_seed']=args.set_sampling_seed                                  # sampling seed
if args.boundary_smooth!=None:
    params['boundary_smooth']=args.boundary_smooth                                      # boundary smooth --> standard deviation 
if args.soft_label!=None:
    params['soft_label']=args.soft_label                                                # soft label
if args.log_train_results!=None:
    params['log_train_results']=args.log_train_results                                  # log traib results

# postprocess
if args.postprocess_type!=None:
    params['postprocess']['type']=args.postprocess_type                                 # type of postprocessing
if args.postprocess_value!=None:
    params['postprocess']['value']=args.postprocess_value                               # value for the postprocessing


# make directory for configurations

if not os.path.exists('configs'):
    os.makedirs('configs')
    


# write the json file

file_name = os.path.join('configs', f'{params["naming"]}.json')
with open(file_name, 'w') as outfile:
    json.dump(params, outfile, ensure_ascii=False)


# split_num = 4

# for split_id in range(1, split_num+1):
    
#     params = copy.deepcopy(params_gtea)

#     params['split_id'] = split_id
#     params['naming'] = f'GTEA-Trained-S{split_id}'

#     if not os.path.exists('configs'):
#         os.makedirs('configs')
     
#     file_name = os.path.join('configs', f'{params["naming"]}.json')

#     with open(file_name, 'w') as outfile:
#         json.dump(params, outfile, ensure_ascii=False)


# ###################### 50salads #######################

# split_num = 5

# for split_id in range(1, split_num+1):
    
#     params = copy.deepcopy(params_50salads)

#     params['split_id'] = split_id
#     params['naming'] = f'50salads-Trained-S{split_id}'

#     if not os.path.exists('configs'):
#         os.makedirs('configs')
     
#     file_name = os.path.join('configs', f'{params["naming"]}.json')

#     with open(file_name, 'w') as outfile:
#         json.dump(params, outfile, ensure_ascii=False)

# ###################### Breakfast #######################

# split_num = 4

# for split_id in range(1, split_num+1):
    
#     params = copy.deepcopy(params_breakfast)

#     params['split_id'] = split_id
#     params['naming'] = f'Breakfast-Trained-S{split_id}'

#     if not os.path.exists('configs'):
#         os.makedirs('configs')
     
#     file_name = os.path.join('configs', f'{params["naming"]}.json')

#     with open(file_name, 'w') as outfile:
#         json.dump(params, outfile, ensure_ascii=False)
