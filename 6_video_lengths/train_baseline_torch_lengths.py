# This takes two arguments for the protocol,
# either 'https' or 'tor', and the length to
# truncate the traffic flows down to in seconds
#
# Outputs are the trained models.

import torch
import mymodels_torch
import numpy
import sys

protocol = sys.argv[1]

length = int(sys.argv[2])

INPUT_SHAPES = {'schuster8': (1, length * 8),
                'dschuster16': (2, length * 16)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

def glorot_uniform(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# helpfully provided by ChatGPT
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = numpy.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for representation in ['dschuster16', 'schuster8']:
    #for protocol in ['https', 'tor']:
        try:
            # if they exist, load the data tensors that resulted from raw_to_csv.py,
            # csv_to_pkl.py, csv_to_pkl_open.py, and keras_to_torch_splits.py
            train_tensors = torch.load('../4_open_world_enhancements/' + representation + '_' + protocol + '_train_tensors.pt')
            # truncate the lengths down to the specified number of time steps
            print('Train tensors shape before slicing: ', train_tensors[0].shape)
            train_tensors = (train_tensors[0][:, :, :INPUT_SHAPES[representation][1]], train_tensors[1])
            print('Train tensors shape after slicing: ', train_tensors[0].shape)
            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            val_tensors = torch.load('../4_open_world_enhancements/' + representation + '_' + protocol + '_val_tensors.pt')
            # truncate the lengths down to the specified number of time steps
            val_tensors = (val_tensors[0][:, :, :INPUT_SHAPES[representation][1]], val_tensors[1])
            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size = BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['batch_size'],
                                                       shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size = len(val_dataset),
                                                     shuffle=False)
        except Exception as e:
            # we expect to hit this condition for schuster8_https and dschuster16_tor
            print(e)
            continue
        for trial in range(20):
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation], 61,
                                                BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
            if BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['fc_init'] == 'glorot_uniform':
                model.apply(glorot_uniform)
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['lr'])
            early_stopping = EarlyStopping(patience = 20,
                                           verbose = True,
                                           path = ('models/' + representation + '_' + protocol + '_baseline_model_' + str(length) + '_' + str(trial) + '.pt'))
            print('Starting to train now for', representation, protocol)
            for epoch in range(240):
                model.train()
                training_loss = 0.0
                for x_train, y_train in train_loader:
                    optimizer.zero_grad()
                    outputs = model(x_train.to(device), training = True)
                    loss = criterion(outputs, y_train.to(device))
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        outputs = model(x_val.to(device), training = False)
                        loss = criterion(outputs, y_val.to(device))
                        val_loss += loss.item()

                print(f'Epoch {epoch+1} \t Training Loss: {training_loss / len(train_dataset)} \t Validation Loss: {val_loss / len(val_dataset)}')
                # check if this is a new low validation loss and, if so, save the model
                #
                # otherwise increment the counter towards the patience limit
                early_stopping(val_loss / len(val_dataset), model)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    break
