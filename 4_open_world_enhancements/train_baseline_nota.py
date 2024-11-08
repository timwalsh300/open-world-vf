# This takes arguments for the protocol and lambda_g
# which controls the mix of NOTA and real unmonitored
# training instances. It instantiates the model and loads
# the monitored training data and unmonitored training
# data separately so that we can also experiment with
# different pairings when creating NOTA instances.
#
# Outputs are the trained models

import torch
import mymodels_torch
import numpy
import sys
from sklearn.metrics import precision_recall_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

protocol = sys.argv[1]

plot_tsne = False

# 0.00 = only real unmonitored instances, same as baseline
# 0.25 = overweight real unmonitored instances
# 0.50 = equal weight for real unmonitored and NOTA
# 0.75 = overweight NOTA instances
# 1.00 = only NOTA, for when training data is monitored-only
lambda_g = float(sys.argv[2])

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                        'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
                        
NOTA_HYPERPARAMETERS = {'schuster8_tor': {'eps_fraction': 0.2, 'pgd_steps': 40, 'alpha': 0.05, 'noise_fraction': 0.1},
                        'dschuster16_https': {'eps_fraction': 0.0004, 'pgd_steps': 40, 'alpha': 0.05, 'noise_fraction': 0.00005}}

def pgd_attack(baseline_model, x, y, pgd_steps, eps_fraction):
    overall_std = torch.std(x, unbiased=False)
    eps = overall_std * eps_fraction
    step_size = eps / pgd_steps
    x_adv = x.clone().detach().requires_grad_(True)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(pgd_steps):
        outputs = baseline_model(x_adv, training=False)
        loss = criterion(outputs, y)
        loss.backward()
        with torch.no_grad():
            x_adv -= step_size * x_adv.grad.sign()
            x_adv.clamp_(x - eps, x + eps)
            x_adv.clamp_(x.min(), x.max())
        x_adv.grad.zero_()
    return x_adv.detach()

def generate_NOTA_padding_instances(x_train_mon, number_of_NOTA_instances, baseline_model, pgd_steps, eps_fraction, alpha, noise_fraction):
    if len(x_train_mon) < number_of_NOTA_instances // 2:
        indices = torch.randint(0, len(x_train_mon), (number_of_NOTA_instances // 2,))
        x_train_mon_selected = x_train_mon[indices].detach().requires_grad_(True)
    else:
        x_train_mon_selected = x_train_mon[:number_of_NOTA_instances // 2].detach().requires_grad_(True)
    target_labels = torch.zeros(len(x_train_mon_selected), 61).to(device)
    target_labels[:, 60] = 1.0
    x_adv = pgd_attack(baseline_model, x_train_mon_selected, target_labels, pgd_steps, eps_fraction).detach()
    weight = numpy.random.uniform(alpha, 1 - alpha)
    overall_std = torch.std(x_train_mon, unbiased=False)
    noise = torch.randn_like(x_train_mon_selected) * (overall_std * noise_fraction)
    x_train_NOTA_wavg = weight * x_train_mon_selected + (1 - weight) * x_adv + noise
    x_train_NOTA_mean = 0.5 * x_train_mon_selected + 0.5 * x_adv + noise
    x_train_NOTA = torch.cat([x_train_NOTA_wavg, x_train_NOTA_mean], dim=0)
    torch.clamp_(x_train_NOTA, x_train_mon.min(), x_train_mon.max())
    return x_train_NOTA

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
            #self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func(f'Validation PR-AUC increased ({self.val_loss_min:.6f} --> {val_loss:.6f})...')
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for representation in ['dschuster16', 'schuster8']:
    #for protocol in ['https', 'tor']:
        try:
            # if they exist, load the data tensors that resulted from raw_to_csv.py,
            # csv_to_pkl.py, csv_to_pkl_open.py, and keras_to_torch_splits.py
            train_mon_tensors = torch.load(representation + '_' + protocol + '_train_mon_tensors.pt')
            train_mon_dataset = torch.utils.data.TensorDataset(*train_mon_tensors)
            train_unmon_tensors = torch.load(representation + '_' + protocol + '_train_unmon_tensors.pt')
            train_unmon_dataset = torch.utils.data.TensorDataset(*train_unmon_tensors)
            val_tensors = torch.load(representation + '_' + protocol + '_val_tensors.pt')
            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            # the balance of monitored to unmonitored instances in the full
            # training dataset is about 1:2, so we adjust the batch sizes correspondingly
            # for the monitored and unmonitored dataloaders
            train_mon_loader = torch.utils.data.DataLoader(train_mon_dataset,
                                                       batch_size = int(0.33 * BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['batch_size']),
                                                       shuffle=False)
            # further adjust the number of real unmonitored instances
            # by 1 - lambda_g
            try:
                train_unmon_loader = torch.utils.data.DataLoader(train_unmon_dataset,
                                                       batch_size = int((1 - lambda_g) * 0.67 * BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['batch_size']),
                                                       shuffle=False)
            # this is a workaround for the fact that the batch size can't be 0,
            # but we want to effectively put a weight of 0 on real unmonitored
            # training instances
            except:
                train_unmon_loader = torch.utils.data.DataLoader(train_unmon_dataset,
                                                       batch_size = 1,
                                                       shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size = len(val_dataset),
                                                     shuffle=False)
        except Exception as e:
            # we expect to hit this condition for schuster8_https and dschuster16_tor
            print(e)
            continue
        
        trial = 0
        alpha = NOTA_HYPERPARAMETERS[representation + '_' + protocol]['alpha']
        #noise_fraction = NOTA_HYPERPARAMETERS[representation + '_' + protocol]['noise_fraction']
        #eps_fraction = NOTA_HYPERPARAMETERS[representation + '_' + protocol]['eps_fraction']
        pgd_steps = NOTA_HYPERPARAMETERS[representation + '_' + protocol]['pgd_steps']
        #for trial in range(10):
        for eps_fraction in [0.01, 0.03, 0.05, 0.07, 0.1]:
        #for eps_fraction in [0.00005, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001]:
        # for pgd_steps in [5, 10, 20, 40]:
        #  for alpha in [0.05, 0.2]:
           for noise_fraction in [0.01, 0.03, 0.05, 0.07, 0.1]:
        #  for noise_fraction in [0.00005, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001]:
            # load the pre-trained baseline model which we'll use during PGD
            baseline_model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                         61,
                                                         BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
            baseline_model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model' + str(trial) + '.pt'))
            baseline_model.to(device)
            baseline_model.eval()
            
            # instantiate the model that we'll actually train with NOTA
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                61,
                                                BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
            model.to(device)
            if BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['fc_init'] == 'glorot_uniform':
                model.apply(glorot_uniform)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['lr'])
            early_stopping = EarlyStopping(patience = 20,
                                           verbose = True,
                                           path = (representation + '_' + protocol + '_baseline_nota_model' + str(trial) + '.pt'))
            # these next two lines are used to apply multi-class labels
            # to the NOTA instances
            NOTA_label = torch.zeros(61)
            NOTA_label[60] = 1.0
            print('Starting to train now for', representation, protocol, trial, lambda_g, eps_fraction, pgd_steps, alpha, noise_fraction)
            for epoch in range(240):
                train_mon_iter = iter(train_mon_loader)
                train_unmon_iter = iter(train_unmon_loader)
                # instantiate some lists to hold data
                # for visualization later
                x_batches = []
                y_batches = []
                model.train()
                training_loss = 0.0
                for _ in range(len(train_mon_loader)):
                    optimizer.zero_grad()
                    # get next batches and unpack them
                    train_mon = next(train_mon_iter)
                    x_train_mon, y_train_mon = train_mon
                    x_train_mon = x_train_mon.to(device)
                    # add a 61st class value of 0.0, because the
                    # labels were just 0-59 for the monitored-only task
                    labels_60 = torch.zeros(len(y_train_mon), 1)
                    y_train_mon = torch.cat((y_train_mon, labels_60), dim=1)
                    
                    train_unmon = next(train_unmon_iter)
                    x_train_unmon, y_train_unmon = train_unmon
                    x_train_unmon = x_train_unmon.to(device)
                    
                    # create a number of NOTA instances equal to the batch
                    # size minus the number of real unmonitored instances
                    # while maintaining a ratio of 1:2 real monitored to
                    # real unmonitored + NOTA
                    number_of_NOTA_instances = int(lambda_g * 0.67 * BASELINE_HYPERPARAMETERS[representation + '_' + protocol]['batch_size'])
                    x_train_NOTA = generate_NOTA_padding_instances(x_train_mon, number_of_NOTA_instances, baseline_model, pgd_steps, eps_fraction, alpha, noise_fraction)
                    y_train_NOTA = NOTA_label.repeat(len(x_train_NOTA), 1)
                    
                    # train on real monitored, real unmonitored, and NOTA
                    # all at once so that batch normalization statistics
                    # are calculated the same way that the baseline
                    # model calculates them when lambda_g = 0
                    if lambda_g < 1.0:
                        x_train = torch.cat([x_train_mon, x_train_unmon, x_train_NOTA])
                        y_train = torch.cat([y_train_mon, y_train_unmon, y_train_NOTA])
                    # don't include that one instance from the train_unmon_loader
                    # that we drew as a workaround to prevent an error
                    else:
                        x_train = torch.cat([x_train_mon, x_train_NOTA])
                        y_train = torch.cat([y_train_mon, y_train_NOTA])
                    outputs = model(x_train, training=True)
                    loss = criterion(outputs, y_train.to(device))
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    
                    # produce a t-SNE to show the relationship
                    # between the monitored, unmonitored, and NOTA instances...
                    # start here by saving the instances from each batch for later
                    if epoch == 0 and plot_tsne:
                        x_batches.append(baseline_model.extract_features_flattened(x_train).detach().cpu())
                        # convert convert from one-hot encoding to single labels here too
                        y_batches.append(torch.full((y_train_mon.shape[0],), 0, dtype=torch.float32))
                        y_batches.append(torch.full((y_train_unmon.shape[0],), 1, dtype=torch.float32))
                        # re-label the NOTA instances to 61 so we can distinguish between them and the
                        # real unmonitored instances on the t-SNE plot
                        y_batches.append(torch.full((y_train_NOTA.shape[0],), 2, dtype=torch.float32))

                # check discriminator's performance on NOTA
                # instances and the validation set
                val_loss = 0.0
                with torch.no_grad():
                    model.eval()
                    #output = model(x_train_NOTA, training = False)
                    #preds = torch.softmax(output, dim=1)
                    #preds_binary, _ = torch.max(preds[:, :60], dim=1)
                    #mean_pred_NOTA = preds_binary.detach().cpu().mean()
                
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(device)
                        logits_val = model(x_val, training = False)
                        loss = criterion(logits_val, y_val.to(device))
                        val_loss += loss.item()
                        
                        # compute PR-AUC over the validation set
                        y_val_binary = 1 - y_val[:, 60]
                        preds_val = torch.softmax(logits_val, dim=1)
                        preds_val_binary, _ = torch.max(preds_val[:, :60], dim=1)
                        precisions, recalls, thresholds = precision_recall_curve(y_val_binary.cpu(), preds_val_binary.cpu())
                        pr_auc = auc(recalls, precisions)

                # every few epochs, produce a t-SNE to show the relationship
                # between the monitored, unmonitored, and NOTA instances
                if epoch == 0 and plot_tsne:
                    print('tsne: concatenating batches')
                    x_batches_np_flattened = torch.cat(x_batches).numpy()
                    # we need to flatten it before passing it to the t-SNE algorithm
                    # x_batches_np_flattened = x_batches_np.reshape(x_batches_np.shape[0], -1)
                    # extract_features_flattened() takes care of this
                    y_batches_np = torch.cat(y_batches).numpy()
                    print('tsne: reducing dimensionality')
                    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200)
                    tsne_results = tsne.fit_transform(x_batches_np_flattened)
                    plt.figure(figsize=(16, 12))
                    unique_labels = numpy.unique(y_batches_np)  # Should be [0, 1, 2]
                    colors = ['blue', 'black', 'green']  # Colors for monitored, unmonitored, and NOTA
                    labels_dict = {0: 'Monitored', 1: 'Unmonitored', 2: 'NOTA'}  # Labels for the legend
                    # Plot each category with its own color and label
                    for i, label in enumerate(unique_labels):
                        idxs = numpy.where(y_batches_np == label)
                        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], c=colors[i], label=labels_dict[label], alpha=0.5)
                    plt.title(f't-SNE Visualization of Training Data at Epoch {epoch}')
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.savefig(f'tsne_nota_{protocol}_features_{eps_fraction}_{pgd_steps}_{alpha}_{noise_fraction}.png')
                    plt.close()

                print(f'Epoch {epoch+1} \t Training loss: {training_loss} \t Val Loss: {val_loss / len(val_dataset)} \t Val PR-AUC: {pr_auc}')
                # check if this is a new low validation loss and, if so, save the model
                #
                # otherwise increment the counter towards the patience limit
                #early_stopping(val_loss / len(val_dataset), model)
                early_stopping(-pr_auc, model)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    break
