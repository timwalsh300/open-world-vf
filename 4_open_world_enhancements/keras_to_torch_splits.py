import torch
import pickle

for representation in ['dschuster16', 'schuster8']:
    for protocol in ['https', 'tor']:
        try:
            with open('../3_open_world_baseline/' + representation + '_open_world_' + protocol + '_splits.pkl', 'rb') as handle:
                splits = pickle.load(handle)
                print(splits['x_train'].shape)
                x_train = torch.tensor(splits['x_train'], dtype=torch.float32)
                print(x_train.shape)
                x_train = x_train.transpose(1, 2)
                print(x_train.shape)
                y_train = torch.tensor(splits['y_train'], dtype=torch.float32)
                train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
                torch.save(train_dataset.tensors, representation + '_' + protocol + '_train_tensors.pt')                

                x_val = torch.tensor(splits['x_val'], dtype=torch.float32)
                x_val = x_val.transpose(1, 2)
                y_val = torch.tensor(splits['y_val'], dtype=torch.float32)
                val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
                torch.save(val_dataset.tensors, representation + '_' + protocol + '_val_tensors.pt')                

                x_test = torch.tensor(splits['x_test_64000'], dtype=torch.float32)
                x_test = x_test.transpose(1, 2)
                y_test = torch.tensor(splits['y_test_64000'], dtype=torch.float32)
                test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
                torch.save(test_dataset.tensors, representation + '_' + protocol + '_test_tensors.pt')                
        except Exception as e:
            print(e)
