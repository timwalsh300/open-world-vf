# This takes no arguments on the command line
#
# Outputs are the trained models.

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
            #self.trace_func(f'Validation PR-AUC increased ({self.val_loss_min:.6f} --> {val_loss:.6f})...')
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
            train_tensors = torch.load(representation + '_' + protocol + '_train_tensors.pt')
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
        for trial in range(10):
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                61,
                                                BEST_HYPERPARAMETERS[representation + '_' + protocol])
            model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model' + str(trial) + '.pt'))
            model.to(device)
            model.eval()     
            netD = mymodels_torch.Discriminator()
            netD.to(device)
            optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0001 / 1.5, betas=(0.5, 0.999))     
            criterion = torch.nn.BCELoss()
            netG = mymodels_torch.Generator()
            netG.to(device)
            optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
            early_stopping = EarlyStopping(patience = 20,
                                           verbose = True,
                                           path = (representation + '_' + protocol + '_baseline_opengan_model' + str(trial) + '.pt'))
            print('Starting to train now for', representation, protocol)
            for epoch in range(240):
                lossD_real = 0.0
                lossD_fake = 0.0
                lossG = 0.0
                for x_train, y_train in train_loader:
                    # train discriminator on real monitored vs. unmonitored + fake monitored
                    optimizerD.zero_grad()
                    netD.train()
                    netG.eval()
                    features_real = model.extract_features(x_train.to(device))
                    output_real = netD(features_real)
                    y_real = 1 - y_train[:, 60]
                    # use the above line for Standard Model, and the below line for monitored only
                    #y_real = torch.ones(len(y_train), dtype=torch.float32)
                    loss = criterion(output_real.squeeze(), y_real.to(device))
                    lossD_real += loss.item()
                    loss.backward()
                    
                    noise = torch.randn(len(y_train), 100, 1, 1, device=device)
                    features_fake = netG(noise)
                    output_fake = netD(features_fake.detach())
                    y_fake_zeros = torch.zeros(len(y_train), dtype=torch.float32)
                    loss = criterion(output_fake.squeeze(), y_fake_zeros.to(device))
                    lossD_fake += loss.item()
                    loss.backward()
                    optimizerD.step()
                    
                    # train generator with discriminator's predictions
                    # on fake monitored
                    optimizerG.zero_grad()
                    netD.eval()
                    netG.train()
                    noise = torch.randn(len(y_train), 100, 1, 1, device=device)
                    features_fake = netG(noise)
                    output_fake = netD(features_fake)
                    y_fake_ones = torch.ones(len(y_train), dtype=torch.float32)
                    loss = criterion(output_fake.squeeze(), y_fake_ones.to(device))
                    lossG += loss.item()
                    loss.backward()
                    optimizerG.step()

                # check discriminator's performance on validation set
                # monitored vs. unmonitored
                val_loss = 0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        netD.eval()
                        features_val = model.extract_features(x_val.to(device))
                        output_val = netD(features_val).squeeze()
                        y_val = 1 - y_val[:, 60]
                        loss = criterion(output_val, y_val.to(device))
                        val_loss += loss.item()
                        
                        # compute PR-AUC over the validation set
                        precisions, recalls, thresholds = precision_recall_curve(y_val.cpu(), output_val.cpu())
                        pr_auc = auc(recalls, precisions)

                print(f'Epoch {epoch+1} \t D_real Loss: {lossD_real} \t D_fake Loss: {lossD_fake} \t G Loss: {lossG} \t Val Loss: {val_loss} \t Val PR-AUC', pr_auc)
                # check if this is a new low validation loss and, if so, save the discriminator
                #
                # otherwise increment the counter towards the patience limit
                #early_stopping(val_loss, netD)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    break