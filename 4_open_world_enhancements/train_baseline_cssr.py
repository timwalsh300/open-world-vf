# This takes no arguments on the command line
#
# Outputs are the trained models.

import torch
import mymodels_torch
import numpy
import sys

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}
                
protocol = sys.argv[1]

# we manually copy and paste these hyperparameters from the output of search.py
BASELINE_MONITORED_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.00014940538775594852, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 6.0487458111998665e-05, 'batch_size': 32}}

# helpfully provided by ChatGPT
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path_backbone='backbone_checkpoint.pt', path_classifier='classifier_checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = numpy.Inf
        self.delta = delta
        self.path_backbone = path_backbone
        self.path_classifier = path_classifier
        self.trace_func = trace_func
    def __call__(self, val_loss, backbone, classifier):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, backbone, classifier)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, backbone, classifier)
            self.counter = 0
    def save_checkpoint(self, val_loss, backbone, classifier):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models...')
            #self.trace_func(f'Validation acc increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models...')
        torch.save(backbone.state_dict(), self.path_backbone)
        torch.save(classifier.state_dict(), self.path_classifier)
        self.val_loss_min = val_loss

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
            val_tensors = torch.load(representation + '_' + protocol + '_val_mon_tensors.pt')
            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size = BASELINE_MONITORED_HYPERPARAMETERS[representation + '_' + protocol]['batch_size'],
                                                       shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size = len(val_dataset),
                                                     shuffle=False)
        except Exception as e:
            # we expect to hit this condition for schuster8_https and dschuster16_tor
            print(e)
            continue
        #for lr in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        #    print('...lr', lr)
        #for gamma in [0.4, 0.2, 0.1, 0.05]:
        #    print('...gamma', gamma)
        for trial in range(10):
            print('...trial', trial)
            # instantiate the feature extraction layers
            backbone = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                   60,
                                                   BASELINE_MONITORED_HYPERPARAMETERS[representation + '_' + protocol])
            backbone.to(device)
            # instantiate the CSSR classifier
            classifier = mymodels_torch.CSSRClassifier(in_channels=256, num_classes=60, hidden_layers=[], latent_channels=32, gamma=0.1)
            classifier.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=BASELINE_MONITORED_HYPERPARAMETERS[representation + '_' + protocol]['lr'])
            early_stopping = EarlyStopping(patience = 60,
                                           verbose = True,
                                           path_backbone = (representation + '_' + protocol + '_baseline_cssr_backbone_model' + str(trial) + '.pt'),
                                           path_classifier = (representation + '_' + protocol + '_baseline_cssr_classifier_model' + str(trial) + '.pt'))
            print('Starting to train now for', representation, protocol)
            for epoch in range(240):
                backbone.train()
                classifier.train()
                training_loss = 0.0
                for x_train, y_train in train_loader:
                    optimizer.zero_grad()
                    x_train_features = backbone.extract_features(x_train.to(device))
                    output = classifier(x_train_features)
                    loss = criterion(output, y_train.to(device))
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                val_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                backbone.eval()
                classifier.eval()
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val_features = backbone.extract_features(x_val.to(device))
                        y_val_indices = torch.argmax(y_val, dim=1)
                        output = classifier(x_val_features)
                        loss = criterion(output, y_val.to(device))
                        val_loss += loss.item()
        
                        # Calculate accuracy
                        predicted_classes = torch.argmax(output, dim=1)
                        correct_predictions += (predicted_classes == y_val_indices.to(device)).sum().item()
                        total_predictions += y_val_indices.size(0)

                # Calculate and print accuracy
                accuracy = correct_predictions / total_predictions
                print(f'Epoch {epoch+1} \t Training Loss: {training_loss / len(train_dataset)} \t Validation Loss: {val_loss / len(val_dataset)} \t Accuracy: {accuracy:.4f}')
                # check if this is a new low validation loss and, if so, save the models
                #
                # otherwise increment the counter towards the patience limit
                early_stopping(val_loss / len(val_dataset), backbone, classifier)
                #early_stopping(-accuracy, classifier)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    break
