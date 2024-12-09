# This takes no arguments on the command line
#
# Outputs are the best trained model for HTTPS-only
# and the best trained model for Tor using Spike
# and Slab Dropout and Concrete Dropout layers,
# and using the original Mixup approach
# to gain adversarial robustness, tuned over
# a range of values for alpha

import torch
import mymodels_torch
import numpy
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import sys

protocol = sys.argv[1]

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
# and we've also appended alpha to this dictionary after tuning between 0.01 and 0.5
BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128, 'alpha': 0.1},
                        'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64, 'alpha': 0.05}}

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
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})...')
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
            train_tensors = torch.load(representation + '_' + protocol + '_train_tensors.pt')
            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            val_tensors = torch.load(representation + '_' + protocol + '_val_tensors.pt')
            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size = BEST_HYPERPARAMETERS[representation + '_' + protocol]['batch_size'],
                                                       shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size = 64,
                                                     shuffle=False)
        except Exception as e:
            # we expect to hit this condition for schuster8_https and dschuster16_tor
            continue

        #global_val_loss_min = numpy.Inf
        #trial = 0
        #for alpha in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]:
        for trial in range(20):
            global_val_loss_min = numpy.Inf
            model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation], 61,
                                                  BEST_HYPERPARAMETERS[representation + '_' + protocol],
                                                  w = 1 / 10 * float(len(train_dataset)),
                                                  d = 1 / float(len(train_dataset)))
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         BEST_HYPERPARAMETERS[representation + '_' + protocol]['lr'])
            early_stopping = EarlyStopping(patience = 20,
                                           verbose = True,
                                           path = (representation + '_' + protocol + '_sscd_mixup_model' + str(trial) + '.pt'),
                                           global_val_loss_min = global_val_loss_min)
            print('Starting to train now for', representation, protocol)
            for epoch in range(240):
                model.train()
                training_loss = 0.0
                for x_train, y_train in train_loader:
                    optimizer.zero_grad()
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                    #lam = numpy.random.beta(alpha, alpha)
                    lam = numpy.random.beta(BEST_HYPERPARAMETERS[representation + '_' + protocol]['alpha'],
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol]['alpha'])
                    # shuffle
                    batch_size = x_train.size(0)
                    index = torch.randperm(batch_size).to(device)
                    # Mixup x_train and y_train
                    mixed_x = lam * x_train + (1 - lam) * x_train[index, :]
                    mixed_y = lam * y_train + (1 - lam) * y_train[index, :]
                    outputs = model(mixed_x, training=True)
                    data_term = criterion(outputs, mixed_y)
                    #print('data term', str(data_term.item()), end = ' ')
                    gaussian_prior_term = get_kl_loss(model) / len(x_train)
                    #print('Gaussian prior term', str(gaussian_prior_term.item()), end = ' ')
                    bernoulli_prior_term = model.bernoulli_kl_loss(representation + '_' + protocol) / len(x_train)
                    #print('Bernoulli prior term', str(bernoulli_prior_term.item()))
                    loss = data_term + gaussian_prior_term + bernoulli_prior_term
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        outputs = model(x_val.to(device))
                        loss = criterion(outputs, y_val.to(device))
                        val_loss += loss.item()

                print(f'Epoch {epoch+1} \t Training Loss: {training_loss / len(train_dataset)} \t Validation Loss: {val_loss / len(val_dataset)}')
                # check if this is a new low validation loss to increment
                # the counter towards the patience limit, but only save the
                # model if this is a new best for any value of the
                # hyperparameter that we're tuning
                early_stopping(val_loss / len(val_dataset), model)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    global_val_loss_min = early_stopping.global_val_loss_min
                    break
            
                #layer_names = ['block1_cd', 'block2_cd', 'block3_cd', 'block4_cd', 'fc1_cd', 'fc2_cd']
                #print('Getting p_drop values...')
                #for name in layer_names:
                #    parameter = getattr(model, name).p
                #    print(f"{name}: {parameter.data}")
