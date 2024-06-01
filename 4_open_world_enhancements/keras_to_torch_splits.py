import torch
import pickle
import numpy

for representation in ['dschuster16', 'schuster8']:
    for protocol in ['https', 'tor']:
        try:
            with open('../3_open_world_baseline/' + representation + '_open_world_' + protocol + '_splits.pkl', 'rb') as handle:
                splits = pickle.load(handle)
                print('x_train', splits['x_train'].shape)
                x_train = torch.tensor(splits['x_train'], dtype=torch.float32)
                print(x_train.shape)
                x_train = x_train.transpose(1, 2)
                print(x_train.shape)
                y_train = torch.tensor(splits['y_train'], dtype=torch.float32)
                train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
                torch.save(train_dataset.tensors, representation + '_' + protocol + '_train_tensors.pt')

                # create another version of x_train, y_train that only contains the
                # monitored class instances, and make the labels just 0-59
                indices = numpy.where(splits['y_train'][:, 60] == 0)[0]
                x_train_mon = splits['x_train'][indices]
                y_train_mon = splits['y_train'][indices]
                y_train_mon = numpy.delete(y_train_mon, 60, axis = 1)
                print('x_train_mon', x_train_mon.shape)
                x_train_mon = torch.tensor(x_train_mon, dtype=torch.float32)
                print(x_train_mon.shape)
                x_train_mon = x_train_mon.transpose(1, 2)
                print(x_train_mon.shape)
                y_train_mon = torch.tensor(y_train_mon, dtype=torch.float32)
                train_mon_dataset = torch.utils.data.TensorDataset(x_train_mon, y_train_mon)
                torch.save(train_mon_dataset.tensors, representation + '_' + protocol + '_train_mon_tensors.pt')

                # create another version of x_train, y_train that only contains the
                # unmonitored class instances, labels are still 0-60
                indices = numpy.where(splits['y_train'][:, 60] == 1)[0]
                x_train_unmon = splits['x_train'][indices]
                y_train_unmon = splits['y_train'][indices]
                print('x_train_unmon', x_train_unmon.shape)
                x_train_unmon = torch.tensor(x_train_unmon, dtype=torch.float32)
                print(x_train_unmon.shape)
                x_train_unmon = x_train_unmon.transpose(1, 2)
                print(x_train_unmon.shape)
                y_train_unmon = torch.tensor(y_train_unmon, dtype=torch.float32)
                train_unmon_dataset = torch.utils.data.TensorDataset(x_train_unmon, y_train_unmon)
                torch.save(train_unmon_dataset.tensors, representation + '_' + protocol + '_train_unmon_tensors.pt')

                print('x_val', splits['x_val'].shape)
                x_val = torch.tensor(splits['x_val'], dtype=torch.float32)
                print(x_val.shape)
                x_val = x_val.transpose(1, 2)
                print(x_val.shape)
                y_val = torch.tensor(splits['y_val'], dtype=torch.float32)
                val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
                torch.save(val_dataset.tensors, representation + '_' + protocol + '_val_tensors.pt')

                # create another version of x_val, y_val that only contains the
                # monitored class instances, and make the labels just 0-59
                indices = numpy.where(splits['y_val'][:, 60] == 0)[0]
                x_val_mon = splits['x_val'][indices]
                y_val_mon = splits['y_val'][indices]
                y_val_mon = numpy.delete(y_val_mon, 60, axis = 1)
                print('x_val_mon', x_val_mon.shape)
                x_val_mon = torch.tensor(x_val_mon, dtype=torch.float32)
                print(x_val_mon.shape)
                x_val_mon = x_val_mon.transpose(1, 2)
                print(x_val_mon.shape)
                y_val_mon = torch.tensor(y_val_mon, dtype=torch.float32)
                val_mon_dataset = torch.utils.data.TensorDataset(x_val_mon, y_val_mon)
                torch.save(val_mon_dataset.tensors, representation + '_' + protocol + '_val_mon_tensors.pt')

                # create another version of x_val, y_val that only contains the
                # unmonitored class instances, labels are still 0-60
                indices = numpy.where(splits['y_val'][:, 60] == 1)[0]
                x_val_unmon = splits['x_val'][indices]
                y_val_unmon = splits['y_val'][indices]
                print('x_val_unmon', x_val_unmon.shape)
                x_val_unmon = torch.tensor(x_val_unmon, dtype=torch.float32)
                print(x_val_unmon.shape)
                x_val_unmon = x_val_unmon.transpose(1, 2)
                print(x_val_unmon.shape)
                y_val_unmon = torch.tensor(y_val_unmon, dtype=torch.float32)
                val_unmon_dataset = torch.utils.data.TensorDataset(x_val_unmon, y_val_unmon)
                torch.save(val_unmon_dataset.tensors, representation + '_' + protocol + '_val_unmon_tensors.pt')

                print('x_test', splits['x_test_64000'].shape)
                x_test = torch.tensor(splits['x_test_64000'], dtype=torch.float32)
                print(x_test.shape)
                x_test = x_test.transpose(1, 2)
                print(x_test.shape)
                y_test = torch.tensor(splits['y_test_64000'], dtype=torch.float32)
                test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
                torch.save(test_dataset.tensors, representation + '_' + protocol + '_test_tensors.pt')
        except Exception as e:
            print(e)
