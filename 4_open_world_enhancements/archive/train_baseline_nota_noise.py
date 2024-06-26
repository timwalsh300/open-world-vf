# This takes either 'tor' or 'https' an argument on
# the command line to split up the training into
# two batch jobs
#
# Outputs are the best trained models for HTTPS-only
# and Tor using the uniform and mean NOTA approach
# to gain adversarial robustness
#
# This one uses Standard Model training data

import torch
import mymodels_torch
import numpy
import sys
from sklearn.metrics import precision_recall_curve, auc

protocol = sys.argv[1]

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                        'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

# helpfully provided by ChatGPT, and now modified to support tuning hyperparameters
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt', trace_func=print, global_val_loss_min = numpy.Inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.global_val_loss_min = global_val_loss_min
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
            self.trace_func(f'Validation PR-AUC increased ({self.val_loss_min:.6f} --> {val_loss:.6f})...')
        self.val_loss_min = val_loss
        if val_loss < global_val_loss_min:
            self.trace_func('Saving new global best model...')
            self.global_val_loss_min = val_loss
            torch.save(model.state_dict(), self.path)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for representation in ['dschuster16', 'schuster8']:
    #for protocol in ['https', 'tor']:
        try:
            # if they exist, load the data tensors that resulted from raw_to_csv.py,
            # csv_to_pkl.py, csv_to_pkl_open.py, and keras_to_torch_splits.py
            train_tensors = torch.load(representation + '_' + protocol + '_train_mon_tensors.pt')
            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            val_tensors = torch.load(representation + '_' + protocol + '_val_tensors.pt')
            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size = BEST_HYPERPARAMETERS[representation + '_' + protocol]['batch_size'],
                                                       shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size = len(val_dataset),
                                                     shuffle=False)
                
        except Exception as e:
            # we expect to hit this condition for schuster8_https and dschuster16_tor
            print(e)
            continue

        # these are constants used to check and set labels after
        # creating the NOTA instances
        nota_label = torch.zeros(61, device = device)
        nota_label[60] = 1.0
        one_tensor = torch.tensor(1.0, device=device)
        global_val_loss_min = numpy.Inf
        for noise_fraction in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        #for trial in range(10):
            #global_val_loss_min = numpy.Inf
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation], 61,
                                                  BEST_HYPERPARAMETERS[representation + '_' + protocol])
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         BEST_HYPERPARAMETERS[representation + '_' + protocol]['lr'])
            early_stopping = EarlyStopping(patience = 20,
                                           verbose = True,
                                           #path = (representation + '_' + protocol + '_baseline_nota_model' + str(trial) + '.pt'),
                                           path = (representation + '_' + protocol + '_baseline_nota_noise_model.pt'),
                                           global_val_loss_min = global_val_loss_min)
            print('Starting to train now for', representation, protocol)
            print('Noise fraction', noise_fraction)
            for epoch in range(240):
                model.train()
                training_loss = 0.0
                for x_train, y_train in train_loader:
                    optimizer.zero_grad()
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                    n1_labels = torch.zeros(len(y_train), 1, device = device)
                    y_train = torch.cat((y_train, n1_labels), dim=1)
                    
                    # prepare 3x NOTA noise instances...
                    overall_std = torch.std(x_train)
                    noise = torch.randn_like(x_train) * (overall_std * noise_fraction)
                    noisy_x1 = x_train + noise
                    noisy_y1 = nota_label.repeat(len(y_train), 1)
                    
                    noise = torch.randn_like(x_train) * (overall_std * noise_fraction)
                    noisy_x2 = x_train + noise
                    noisy_y2 = nota_label.repeat(len(y_train), 1)
                    
                    noise = torch.randn_like(x_train) * (overall_std * noise_fraction)
                    noisy_x3 = x_train + noise
                    noisy_y3 = nota_label.repeat(len(y_train), 1)
                    
                    # concatenate the original x and y tensors with the new
                    # NOTA noise padding instances
                    combined_x = torch.cat([x_train, noisy_x1, noisy_x2, noisy_x3], dim=0)
                    combined_y = torch.cat([y_train, noisy_y1, noisy_y2, noisy_y3], dim=0)
                    
                    outputs = model(combined_x, training=True)
                    loss = criterion(outputs, combined_y)
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        outputs = model(x_val.to(device), training = False)
                        loss = criterion(outputs, y_val.to(device))
                        
                        # compute PR-AUC over the validation set
                        y_val_np = y_val.numpy()
                        true_labels_val = numpy.argmax(y_val_np, axis = 1)
                        true_binary_val = (true_labels_val < 60)
                        val_loss += loss.item()
                        preds = torch.softmax(outputs, dim=1).to('cpu').detach().numpy()
                        scores = []
                        for i in range(len(preds)):
                            scores.append(max(preds[i][:60]))
                        precisions, recalls, thresholds = precision_recall_curve(true_binary_val, scores)
                        pr_auc = auc(recalls, precisions)

                print(f'Epoch {epoch+1} \t Training Loss: {training_loss / (4 * len(train_dataset))} \t Validation Loss: {val_loss / len(val_dataset)} Validation PR-AUC', pr_auc)
                # check if this is a new low validation loss to increment
                # the counter towards the patience limit, but only save the
                # model if this is a new best for any value of the
                # hyperparameter that we're tuning
                early_stopping(-pr_auc, model)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    global_val_loss_min = early_stopping.global_val_loss_min
                    break
