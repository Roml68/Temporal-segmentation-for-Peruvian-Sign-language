import os
import struct
import pandas as pd
from google.protobuf.json_format import MessageToDict
from tensorboard.compat.proto.event_pb2 import Event
from tabulate import tabulate
import re

# Initialize a list to store event data
data = []

root = "/media/iot/ML/Summy/final_version_organized/DiffAct/train_layers_last_layer/manejar_conflictos"

def read_record(f):
    header = f.read(12)
    if len(header) < 12:
        return None

    length, = struct.unpack('Q', header[:8])
    event_data = f.read(length)
    f.read(4)  # Skip CRC checksum

    event = Event()
    event.ParseFromString(event_data)
    return event

def get_folders_name_starting_with(name, directory):
    return [
        item for item in os.listdir(directory)
        if item.startswith(name) and os.path.isdir(os.path.join(directory, item))
    ]

def get_summary_file_name(parent_folder):
    for item in os.listdir(parent_folder):
        if item.startswith('events.out.tfevents.'):
            return item
    return None

def get_best_epoch(df, f1b_col, f1s_col, best_epochs_df_train,best_epochs_df_val, aux_df_train,aux_df_val):

    df.columns = df.columns.str.strip() 


    f1b_train='Train_vs_Test/Train F1B per epoch'
    f1b_val='Train_vs_Test/Test F1S per epoch'
    f1s_train='Train_vs_Test/Train F1S per epoch'
    f1s_val='Train_vs_Test/Test F1S per epoch'





    df['min_train_f1'] = df.apply(lambda row: min(row[f1b_train], row[f1s_train]), axis=1)
    df['min_val_f1'] = df.apply(lambda row: min(row[f1b_val], row[f1s_val]), axis=1)
    
    max_min_f1_train = df['min_train_f1'].max()
    max_min_f1_val = df['min_val_f1'].max()

    candidates_train = df['min_train_f1'] == max_min_f1_train
    candidates_val = df['min_val_f1'] == max_min_f1_val 



    filtered_train = df[candidates_train]
    filtered_val = df[candidates_val]

    # Handle the filtered_train candidates
    if len(filtered_train) > 1:
        filtered_train['max_f1'] = filtered_train[[f1b_train, f1s_train]].max(axis=1)
        best_row_train = filtered_train.loc[filtered_train['max_f1'].idxmax()]
    else:
        best_row_train = filtered_train.iloc[0]

    # Handle the filtered_val candidates
    if len(filtered_val) > 1:
        filtered_val['max_f1'] = filtered_val[[f1b_val, f1s_val]].max(axis=1)
        best_row_val = filtered_val.loc[filtered_val['max_f1'].idxmax()]
    else:
        best_row_val = filtered_val.iloc[0]


    # Convert the best row to a dictionary and update aux_df

    selected_columns_train = ['step', 'Train_vs_Test/Train F1B per epoch', 'Train_vs_Test/Train F1S per epoch','Total Train loss (pondered) per epoch']
    selected_columns_val = ['step', 'Train_vs_Test/Test F1B per epoch', 'Train_vs_Test/Test F1S per epoch','Total Test loss (pondered) per epoch','Test-decoder-agg-Acc','Test-decoder-agg-F1@50']


    # Filter the rows to only include the selected columns
    best_row_train_dict = best_row_train[selected_columns_train].to_dict()
    best_row_val_dict = best_row_val[selected_columns_val].to_dict()

    # Update the auxiliary DataFrames with the filtered dictionaries
    aux_df_train.update(best_row_train_dict)
    aux_df_val.update(best_row_val_dict)


    # Convert aux_df to a DataFrame
    aux_df_train = pd.DataFrame([aux_df_train])
    aux_df_val = pd.DataFrame([aux_df_val])


    print(best_epochs_df_val)



    if not best_epochs_df_train.empty and not best_epochs_df_val.empty:
        best_epochs_df_train = pd.concat([best_epochs_df_train, aux_df_train], ignore_index=True)
        best_epochs_df_val = pd.concat([best_epochs_df_val, aux_df_val], ignore_index=True)

    else:
        
        best_epochs_df_train = aux_df_train
        best_epochs_df_val = aux_df_val



    print(tabulate(best_epochs_df_train, headers='keys', tablefmt='pipe'))
    print(tabulate(best_epochs_df_val, headers='keys', tablefmt='pipe'))

    return best_epochs_df_train,best_epochs_df_val

train_best_epochs_df = pd.DataFrame()
val_best_epochs_df = pd.DataFrame()


folders = get_folders_name_starting_with('break_decoder_12_256_0', root)

print(folders)

for folder in folders:
    data.clear()  # Clear old data for each folder

    aux_train_best_epochs_df = {}
    aux_val_best_epochs_df = {}

    numbers = re.findall(r'\d+', folder)
    aux_train_best_epochs_df.update({'name': folder, 'num_layers': int(numbers[0]), 'num_f_maps': int(numbers[1])})
    aux_val_best_epochs_df.update({'name': folder, 'num_layers': int(numbers[0]), 'num_f_maps': int(numbers[1])})

    name_of_file = get_summary_file_name(os.path.join(root, folder))
    if not name_of_file:
        continue

    complete_path = os.path.join(root, folder, name_of_file)

    with open(complete_path, 'rb') as f:
        while True:
            event = read_record(f)
            if event is None:
                break

            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        data.append({
                            'step': event.step,
                            'tag': value.tag,
                            'value': value.simple_value
                        })

    df = pd.DataFrame(data)
    df_pivot = df.pivot(index='step', columns='tag', values='value').reset_index().dropna()

    # print(df)
    # print("prueba----------------")
    print(df_pivot)
    # print(df_pivot.columns)

    output_dir_complete="/media/iot/ML/Summy/final_version_organized/DiffAct/train_layers_last_layer/manejar_conflictos/complete_results"


    df_pivot.to_json(os.path.join(output_dir_complete,folder+'.json'), orient='records', lines=True)




    # train_best_epochs_df,val_best_epochs_df = get_best_epoch(
    #     df_pivot, f1b_col='F1B per epoch', f1s_col='F1S per epoch',
    #     best_epochs_df_train=train_best_epochs_df,best_epochs_df_val=val_best_epochs_df, 
    #     aux_df_train=aux_train_best_epochs_df,aux_df_val=aux_val_best_epochs_df
    # )


# output_dir="/media/iot/ML/Summy/final_version_organized/DiffAct/train_layers_last_layer/manejar_conflictos/table_results_iterations"
# train_best_epochs_df.to_json(os.path.join(output_dir,'train_table'), orient='records', lines=True)
# val_best_epochs_df.to_json(os.path.join(output_dir,'val_table'), orient='records', lines=True)

