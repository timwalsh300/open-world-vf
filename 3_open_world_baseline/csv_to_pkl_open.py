import pandas
import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical
import pickle
import os

# This is a helper function for open_split() that converts
# a dataframe into x, y numpy arrays in the appropriate forms
def get_x_y(df, data_points,
            target_label,
            target_label_map,
            next_columns):
    x = df.drop(next_columns, axis = 1).values.reshape(df.shape[0], data_points, 1)
    y_strings = df[target_label]
    y = to_categorical(y_strings.map(target_label_map), len(target_label_map))
    return x, y

# This function reads in monitored and unmonitored .csv files, combines them,
# splits them into numpy arrays for training, validation, and testing for our
# open-world scenarios, and then pickles a dictionary of those arrays.
def open_split(monitored_csv_path, 
               unmonitored_csv_path,
               data_points,
               target_label,
               target_label_map,
               output_name):
    # Read in the .csv files providing indices as column names for the
    # packets / periods of time (points of data) followed by the six
    # labels, specifying that the points of data
    # should be 32-bit floats to save memory
    first_columns = [str(i) for i in range(data_points)]
    next_columns = ['region',
                    'heavy_hitter',
                    'platform',
                    'genre',
                    'points',
                    'visit',
                    'id']
    column_names = first_columns + next_columns
    dtype_dict = {str(i): numpy.float32 for i in range(data_points)}
    
    # start by separating the monitored set into parts
    print('reading ' + monitored_csv_path + ' into a dataframe')
    dataframe_list = []
    for chunk in pandas.read_csv(monitored_csv_path,
                                 header = None,
                                 names = column_names,
                                 dtype = dtype_dict,
                                 chunksize = 5000):
        dataframe_list.append(chunk)
    monitored = pandas.concat(dataframe_list, ignore_index = True)
    # shuffle
    monitored = monitored.sample(frac = 1).reset_index(drop = True)
    # find only the us-west-2 (Oregon) instances
    monitored_oregon = monitored[monitored['region'] == 'oregon']
    # get the first 10 instances of each of the 60 videos to put
    # in our test sets
    grouped = monitored_oregon.groupby(target_label)
    monitored_test = pandas.DataFrame()
    for name, group in grouped:
        temp = group.head(10)
        monitored_test = pandas.concat([monitored_test, temp])
    # remove those 600 test set instances from the original dataframe,
    # so we're left with a dataframe to split into training and validation
    monitored_train_val = monitored.drop(monitored_test.index)
    print('monitored dataframe shape is', monitored.shape,
          '\noregon monitored dataframe shape is', monitored_oregon.shape,
          '\nmonitored test dataframe shape is', monitored_test.shape,
          '\nremaining monitored dataframe shape is', monitored_train_val.shape)
    
    # continue with the unmonitored set
    print('reading ' + unmonitored_csv_path + ' into a dataframe')
    dataframe_list = []
    for chunk in pandas.read_csv(unmonitored_csv_path,
                                 header = None,
                                 names = column_names,
                                 dtype = dtype_dict,
                                 chunksize = 5000):
        dataframe_list.append(chunk)
    unmonitored = pandas.concat(dataframe_list, ignore_index = True)
    # shuffle
    unmonitored = unmonitored.sample(frac = 1).reset_index(drop = True)
    # get the (overlapping) sets of test instances for each world size
    unmonitored_1000 = unmonitored.iloc[:1000]
    unmonitored_2000 = unmonitored.iloc[:2000]
    unmonitored_4000 = unmonitored.iloc[:4000]
    unmonitored_8000 = unmonitored.iloc[:8000]
    unmonitored_16000 = unmonitored.iloc[:16000]
    unmonitored_32000 = unmonitored.iloc[:32000]
    unmonitored_64000 = unmonitored.iloc[:64000]
    # this one has what is leftover to use for training and validation
    unmonitored_train_val = unmonitored.iloc[64000:]
    print('unmonitored dataframe shape is', unmonitored.shape,
          '\nunmonitored_1000 shape is', unmonitored_1000.shape,
          '\nunmonitored_2000 shape is', unmonitored_2000.shape,
          '\nunmonitored_4000 shape is', unmonitored_4000.shape,
          '\nunmonitored_8000 shape is', unmonitored_8000.shape,
          '\nunmonitored_16000 shape is', unmonitored_16000.shape,
          '\nunmonitored_32000 shape is', unmonitored_32000.shape,
          '\nunmonitored_64000 shape is', unmonitored_64000.shape,
          '\nremaining unmonitored dataframe shape is', unmonitored_train_val.shape)

    # combine the monitored and unmonitored parts into the final training,
    # validation, and test sets
    train_val_dataframe = pandas.concat([monitored_train_val, unmonitored_train_val], ignore_index = True)
    train_val_split = StratifiedShuffleSplit(n_splits = 1,
                                             test_size = 0.15,
                                             random_state = 42)
    for train_index, val_index in train_val_split.split(train_val_dataframe, train_val_dataframe[target_label]):
        train = train_val_dataframe.iloc[train_index]
        val = train_val_dataframe.iloc[val_index]
    print('training dataframe shape is', train.shape,
          '\nvalidation dataframe shape is', val.shape)
          
    output = {}    
    x_train, y_train = get_x_y(train, data_points, target_label, target_label_map, next_columns)
    output['x_train'] = x_train
    output['y_train'] = y_train
    
    x_val, y_val = get_x_y(val, data_points, target_label, target_label_map, next_columns)
    output['x_val'] = x_val
    output['y_val'] = y_val
          
    test_1000 =  pandas.concat([monitored_test, unmonitored_1000], ignore_index = True)
    print('test_1000 dataframe shape is', test_1000.shape)
    x_test_1000, y_test_1000 = get_x_y(test_1000, data_points, target_label, target_label_map, next_columns)
    output['x_test_1000'] = x_test_1000
    output['y_test_1000'] = y_test_1000
    
    test_2000 =  pandas.concat([monitored_test, unmonitored_2000], ignore_index = True)
    print('test_2000 dataframe shape is', test_2000.shape)
    x_test_2000, y_test_2000 = get_x_y(test_2000, data_points, target_label, target_label_map, next_columns)
    output['x_test_2000'] = x_test_2000
    output['y_test_2000'] = y_test_2000
    
    test_4000 =  pandas.concat([monitored_test, unmonitored_4000], ignore_index = True)
    print('test_4000 dataframe shape is', test_4000.shape)
    x_test_4000, y_test_4000 = get_x_y(test_4000, data_points, target_label, target_label_map, next_columns)
    output['x_test_4000'] = x_test_4000
    output['y_test_4000'] = y_test_4000
    
    test_8000 =  pandas.concat([monitored_test, unmonitored_8000], ignore_index = True)
    print('test_8000 dataframe shape is', test_8000.shape)
    x_test_8000, y_test_8000 = get_x_y(test_8000, data_points, target_label, target_label_map, next_columns)
    output['x_test_8000'] = x_test_8000
    output['y_test_8000'] = y_test_8000
    
    test_16000 =  pandas.concat([monitored_test, unmonitored_16000], ignore_index = True)
    print('test_16000 dataframe shape is', test_16000.shape)
    x_test_16000, y_test_16000 = get_x_y(test_16000, data_points, target_label, target_label_map, next_columns)
    output['x_test_16000'] = x_test_16000
    output['y_test_16000'] = y_test_16000
    
    test_32000 =  pandas.concat([monitored_test, unmonitored_32000], ignore_index = True)
    print('test_32000 dataframe shape is', test_32000.shape)
    x_test_32000, y_test_32000 = get_x_y(test_32000, data_points, target_label, target_label_map, next_columns)
    output['x_test_32000'] = x_test_32000
    output['y_test_32000'] = y_test_32000
    
    test_64000 =  pandas.concat([monitored_test, unmonitored_64000], ignore_index = True)
    print('test_64000 dataframe shape is', test_64000.shape)
    x_test_64000, y_test_64000 = get_x_y(test_64000, data_points, target_label, target_label_map, next_columns)
    output['x_test_64000'] = x_test_64000
    output['y_test_64000'] = y_test_64000
    
    with open(output_name, 'wb') as handle:
        pickle.dump(output, handle)

    print('finished writing ' + output_name)

# main
base_path = '/home/timothy.walsh/VF/'
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
id_map = {}

for i in range(20,30):
    id_map[i] = i - 20
for i in range(60,70):
    id_map[i] = i - 50
for i in range(100,110):
    id_map[i] = i - 80
for i in range(140,150):
    id_map[i] = i - 110
for i in range(180,190):
    id_map[i] = i - 140
for i in range(220,230):
    id_map[i] = i - 170
id_map[240] = 60

print(id_map)

#for representation in ['sirinam_wf', 'sirinam_vf', 'rahman', 'hayden', 'schuster2', 'schuster4', 'schuster8', 'dschuster8', 'schuster16', 'dschuster16']:
for representation in ['schuster8']:
    for protocol in ['https', 'tor']:
        # remember that we're only using Vimeo instances now
        monitored_csv_path = (base_path + '1_csv_to_pkl/' + representation +
                              '_monitored_' + protocol + '_vimeo.csv')
        unmonitored_csv_path = (base_path + '3_open_world_baseline/' +
                                representation + '_unmonitored_' + protocol + '.csv')
#        os.system('cat ' + base_path + '0_raw_to_csv/' + representation + 
#                  '/unmonitored_' + protocol + '/* > ' + unmonitored_csv_path)
#        print('finished writing ' + unmonitored_csv_path)
        open_split(monitored_csv_path,
                   unmonitored_csv_path,
                   data_points_map[representation],
                   'id',
                   id_map,
                   base_path + '3_open_world_baseline/' + representation + '_open_world_' + protocol + '.pkl')
