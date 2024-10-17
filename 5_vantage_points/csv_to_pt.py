import pandas
import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical
import torch
import os

# This function reads in a .csv file, splits it into PyTorch tensors for train, val
# and test splits for a closed-world scenario, and then saves the tensors
def closed_split(input_csv, data_points, target_label, target_label_map, output_name, dschuster):
    # Read in the .csv file providing indices as column names for the
    # packets / periods of time (points of data) followed by the six
    # labels, specifying that the points of data
    # should be 32-bit floats to save memory
    first_columns = [str(i) for i in range(data_points)]
    next_columns = ['region', 'heavy_hitter', 'platform', 'genre', 'points', 'visit', 'id']
    column_names = first_columns + next_columns
    dtype_dict = {str(i): numpy.float32 for i in range(data_points)}
    print('reading ' + input_csv + ' into a dataframe', flush = True)
    dataframe_list = []
    for chunk in pandas.read_csv(input_csv,
                                header = None,
                                names = column_names,
                                dtype = dtype_dict,
                                chunksize = 5000):
        dataframe_list.append(chunk)
    dataframe = pandas.concat(dataframe_list, ignore_index = True)

    # Split out 10% as a val set using stratified sampling on the second argument
    # to the split() call
    print('working on test split', flush = True)
    test_split = StratifiedShuffleSplit(n_splits = 1,
                                        test_size = 0.10,
                                        random_state = 42)
    for work_index, test_index in test_split.split(dataframe, dataframe[target_label]):
        work_set = dataframe.iloc[work_index]
        test_set = dataframe.iloc[test_index]
    # print(str(test_set[target_label].value_counts()))

    print('working on val split', flush = True)
    # Split out another 15% as a validation set
    # using the same approach, leaving us with a training set that is 70% of the original
    val_split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.15, random_state = 42)
    for train_index, val_index in val_split.split(work_set, work_set[target_label]):
        train_set = work_set.iloc[train_index]
        val_set = work_set.iloc[val_index]

    print('train, val, test shapes are ' + str(train_set.shape),
          str(val_set.shape), str(test_set.shape), flush = True)

    # insert modification here for dschuster
    if dschuster:
        x_train = train_set.drop(next_columns, axis = 1).values.reshape(train_set.shape[0],
                                                                        int(data_points / 2), 2)        
    else:
        x_train = train_set.drop(next_columns, axis = 1).values.reshape(train_set.shape[0],
                                                                        data_points, 1)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_train = x_train.transpose(1, 2)
    y_train_strings = train_set[target_label]
    y_train = to_categorical(y_train_strings.map(target_label_map), 
                             len(target_label_map))
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    torch.save(train_dataset.tensors, 'train_' + output_name)

    # insert modification here for dschuster
    if dschuster:
        x_val = val_set.drop(next_columns, axis = 1).values.reshape(val_set.shape[0],
                                                                    int(data_points / 2), 2)
    else:
        x_val = val_set.drop(next_columns, axis = 1).values.reshape(val_set.shape[0],
       	       	       	       	       	       	       	       	    data_points, 1)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    x_val = x_val.transpose(1, 2)
    y_val_strings = val_set[target_label]
    y_val = to_categorical(y_val_strings.map(target_label_map), 
                           len(target_label_map))
    y_val = torch.tensor(y_val, dtype=torch.float32)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    torch.save(val_dataset.tensors, 'val_' + output_name)

    # insert modification here for dschuster
    if dschuster:
        x_test = test_set.drop(next_columns, axis = 1).values.reshape(test_set.shape[0],
                                                                      int(data_points / 2), 2)
    else:
        x_test = test_set.drop(next_columns, axis = 1).values.reshape(test_set.shape[0],
                                                                      data_points, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    x_test = x_test.transpose(1, 2)
    y_test_strings = test_set[target_label]
    y_test = to_categorical(y_test_strings.map(target_label_map),
                            len(target_label_map))
    y_test = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    torch.save(test_dataset.tensors, 'test_' + output_name)

    print('finished writing all ' + output_name, flush = True)


# main
data_points_map = {'sirinam_wf': 5000,
                   'sirinam_vf': 25000,
                   'rahman': 25000,
                   'hayden': 25000,
                   'schuster2': 480,
                   'schuster4': 960,
                   'schuster8': 1920,
                   'dschuster8': 3840,
                   'schuster16': 3840,
                   'dschuster16': 7680}
id_map = {'youtube': {},
          'facebook': {},
          'vimeo': {},
          'rumble': {}}

for i in range(0,10):
    id_map['youtube'][i] = i - 0
for i in range(10,20):
    id_map['facebook'][i] = i - 10
for i in range(20,30):
    id_map['vimeo'][i] = i - 20
for i in range(30,40):
    id_map['rumble'][i] = i - 30

for i in range(40,50):
    id_map['youtube'][i] = i - 30
for i in range(50,60):
    id_map['facebook'][i] = i - 40
for i in range(60,70):
    id_map['vimeo'][i] = i - 50
for i in range(70,80):
    id_map['rumble'][i] = i - 60

for i in range(80,90):
    id_map['youtube'][i] = i - 60
for i in range(90,100):
    id_map['facebook'][i] = i - 70
for i in range(100,110):
    id_map['vimeo'][i] = i - 80
for i in range(110,120):
    id_map['rumble'][i] = i - 90

for i in range(120,130):
    id_map['youtube'][i] = i - 90
for i in range(130,140):
    id_map['facebook'][i] = i - 100
for i in range(140,150):
    id_map['vimeo'][i] = i - 110
for i in range(150,160):
    id_map['rumble'][i] = i - 120

for i in range(160,170):
    id_map['youtube'][i] = i - 120
for i in range(170,180):
    id_map['facebook'][i] = i - 130
for i in range(180,190):
    id_map['vimeo'][i] = i - 140
for i in range(190,200):
    id_map['rumble'][i] = i - 150

for i in range(200,210):
    id_map['youtube'][i] = i - 150
for i in range(210,220):
    id_map['facebook'][i] = i - 160
for i in range(220,230):
    id_map['vimeo'][i] = i - 170
for i in range(230,240):
    id_map['rumble'][i] = i - 180

print(id_map)

for representation in ['schuster8', 'dschuster8', 'schuster16', 'dschuster16']:
    dschuster = True if 'dschuster' in representation else False
    for protocol in ['https', 'tor']:
        for platform in ['youtube', 'facebook', 'vimeo', 'rumble']:
            for region in ['africa', 'brazil', 'frankfurt', 'london', 'oregon', 'seoul', 'stockholm', 'sydney', 'uae', 'virginia']:
                big_csv = ('cat ../../0_raw_to_csv/' + representation +
                           '/monitored_' + protocol + '/' + region + '_' + protocol + '.csv')
                small_csv = (representation + '_' + protocol +
                             '_' + platform + '_' + region + '.csv')
                os.system(big_csv + ' | grep ' + platform + ' > ' + small_csv)
                print('finished writing ' + small_csv, flush = True)
                closed_split(small_csv,
                             data_points_map[representation],
                             'id',
                             id_map[platform],
                             small_csv[:-3] + 'pt',
                             dschuster)
