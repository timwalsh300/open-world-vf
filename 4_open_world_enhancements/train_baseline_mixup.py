# This takes no arguments on the command line
#
# Outputs are the best trained models for HTTPS-only
# and Tor using the original Mixup approach
# to gain adversarial robustness, tuned over
# a range of values for alpha
#
# Optionally, it also outputs a t-SNE plot of the
# real monitored and unmonitored training instances
# and mixup virtual instances in feature space

import torch
import mymodels_torch
import numpy
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

#protocol = sys.argv[1]

plot_tsne = False

# we manually copy and paste these hyperparameters from the output of search_open.py
BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                        'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

# after tuning between 0.01 and 0.5
MIXUP_HYPERPARAMETERS = {'schuster8_tor': {'alpha': 0.1},
                        'dschuster16_https': {'alpha': 0.05}}

# helpfully provided by ChatGPT, and now modified to support tuning for alpha
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
    for protocol in ['https', 'tor']:
        try:
            # if they exist, load the data tensors that resulted from raw_to_csv.py,
            # csv_to_pkl.py, csv_to_pkl_open.py, and keras_to_torch_splits.py
            train_tensors = torch.load(representation + '_' + protocol + '_train_tensors.pt')
            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            val_tensors = torch.load(representation + '_' + protocol + '_val_tensors.pt')
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

        #global_val_loss_min = numpy.Inf
        #trial = 0
        #for alpha in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]:
        for trial in range(20):
            global_val_loss_min = numpy.Inf
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation], 61,
                                                  BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['lr'])
            early_stopping = EarlyStopping(patience = 20,
                                           verbose = True,
                                           path = (representation + '_' + protocol + '_baseline_mixup_model' + str(trial) + '.pt'),
                                           global_val_loss_min = global_val_loss_min)
            print('Starting to train now for', representation, protocol)
            for epoch in range(240):
                model.train()
                training_loss = 0.0
                # instantiate some lists to hold data
                # for visualization later
                x_batches = []
                y_batches = []
                for x_train, y_train in train_loader:
                    optimizer.zero_grad()
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                    #lam = numpy.random.beta(alpha, alpha)
                    lam = numpy.random.beta(MIXUP_HYPERPARAMETERS[representation + '_' + protocol]['alpha'],
                                            MIXUP_HYPERPARAMETERS[representation + '_' + protocol]['alpha'])
                    # shuffle
                    batch_size = x_train.size(0)
                    index = torch.randperm(batch_size).to(device)
                    # Mixup x_train and y_train
                    mixed_x = lam * x_train + (1 - lam) * x_train[index, :]
                    mixed_y = lam * y_train + (1 - lam) * y_train[index, :]
                    outputs = model(mixed_x, training=True)
                    loss = criterion(outputs, mixed_y)
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    
                    # produce a t-SNE to show the relationship
                    # between the monitored, unmonitored, and mixup instances...
                    # start here by saving the instances from each batch for later
                    if epoch == 50 and plot_tsne:
                        x_all = torch.cat([x_train, mixed_x])
                        x_batches.append(model.extract_features_flattened(x_all).detach().cpu())
                        # Create labels for real data (assuming y_train are one-hot encoded and index 60 is unmonitored)
                        y_labels_real = torch.where(y_train[:, 60] == 1, 1, 0)  # 1 for unmonitored, 0 for monitored
                        y_batches.append(y_labels_real.cpu())
                        # Create labels for mixup data
                        y_labels_mixed = torch.zeros(mixed_y.shape[0], dtype=torch.float32)
                        for i in range(mixed_y.shape[0]):
                            if mixed_y[i, 60] == 1.0: # mix is between unmonitored instances
                                y_labels_mixed[i] = 1
                            elif torch.max(mixed_y[i, :60]) == 1.0:  # mix is between instances of same monitored class
                                y_labels_mixed[i] = 0
                            else: # mix is between different classes
                                y_labels_mixed[i] = 2
                        y_batches.append(y_labels_mixed.cpu())

                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        outputs = model(x_val.to(device), training = False)
                        loss = criterion(outputs, y_val.to(device))
                        val_loss += loss.item()

                # at the specified epoch, produce a t-SNE plot to show the relationship
                # between the monitored, unmonitored, and mixup instances
                if epoch == 50 and plot_tsne:
                    print('tsne: concatenating batches')
                    x_batches_np_flattened = torch.cat(x_batches).numpy()
                    y_batches_np = torch.cat(y_batches).numpy()
                    print('tsne: reducing dimensionality')
                    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200)
                    tsne_results = tsne.fit_transform(x_batches_np_flattened)
                    plt.figure(figsize=(16, 12))
                    colors = ['blue', 'black', 'green']  # Colors for monitored, unmonitored, and mixup
                    labels_dict = {0: 'Monitored', 1: 'Unmonitored', 2: 'mixup'}  # Labels for the legend
                    # Plot each category with its own color and label
                    for i in [0, 1, 2]:
                        idxs = numpy.where(y_batches_np == i)
                        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], c=colors[i], label=labels_dict[i], alpha = 0.5)
                    plt.title(f't-SNE Visualization of Training Data at Epoch {epoch}')
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.savefig(f'tsne_mixup_{protocol}_features.png')
                    plt.close()

                print(f'Epoch {epoch+1} \t Training Loss: {training_loss / len(train_dataset)} \t Validation Loss: {val_loss / len(val_dataset)}')
                # check if this is a new low validation loss to increment
                # the counter towards the patience limit, but only save the
                # model if this is a new best for any alpha so far
                early_stopping(val_loss / len(val_dataset), model)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    global_val_loss_min = early_stopping.global_val_loss_min
                    break
