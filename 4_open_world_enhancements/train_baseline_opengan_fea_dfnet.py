# This takes arguments for the protocol and lambda_g. It loads
# pre-trained baseline models to use as feature extractors
# and trains a discriminator on real monitored and
# unmonitored class features, plus fake monitored features
# from a generator that is being trained adversarially
#
# The generator used here is an adaptation of the OpenGAN
# code, and the discriminator is the fully-connected layers
# of our DFNet architecture
#
# Outputs are the trained discriminators

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
# 0.50 = equal weight for real unmonitored and fake monitored
# 0.75 = overweight fake monitored instances
# 1.00 = only fake monitored, for when training data is monitored-only
lambda_g = float(sys.argv[2])

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
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
            #self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func(f'Validation PR-AUC increased ({self.val_loss_min:.6f} --> {val_loss:.6f})...')
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
                                                       batch_size = int(0.33 * BEST_HYPERPARAMETERS[representation + '_' + protocol]['batch_size']),
                                                       shuffle=False)
            # further adjust the number of real unmonitored instances
            # by 1 - lambda_g
            try:
                train_unmon_loader = torch.utils.data.DataLoader(train_unmon_dataset,
                                                       batch_size = int((1 - lambda_g) * 0.67 * BEST_HYPERPARAMETERS[representation + '_' + protocol]['batch_size']),
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
        
        #trial = 0
        lambda_fm = 1e-4
        lambda_e = 1
        #for lambda_fm in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        #    print('...lambda_fm =', lambda_fm)
        #for lambda_e in [1, 2, 3, 4, 5]:
        #    print('...lambda_e =', lambda_e)
        for trial in range(20):
            #print('...lambda_g =', lambda_g)
            print('...trial', trial)
            # load the pre-trained model that does feature extraction
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                61,
                                                BEST_HYPERPARAMETERS[representation + '_' + protocol])
            model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model' + str(trial) + '.pt'))
            model.to(device)
            model.eval()     
            # instantiate the discriminator
            netD = mymodels_torch.DiscriminatorDFNet_fea(INPUT_SHAPES[representation],
                                                         61,
                                                         BEST_HYPERPARAMETERS[representation + '_' + protocol])
            if BEST_HYPERPARAMETERS[representation + '_' + protocol]['fc_init'] == 'glorot_uniform':
                netD.apply(glorot_uniform)
            netD.to(device)
            optimizerD = torch.optim.Adam(netD.parameters(),
                                          lr=BEST_HYPERPARAMETERS[representation + '_' + protocol]['lr'])
            criterionD = torch.nn.CrossEntropyLoss()
            # instantiate the generator
            netG = mymodels_torch.GeneratorDFNet_fea(INPUT_SHAPES[representation],
                                                     BEST_HYPERPARAMETERS[representation + '_' + protocol])
            netG.to(device)
            optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001)
            criterionG = torch.nn.BCEWithLogitsLoss()
            early_stopping = EarlyStopping(patience = 20,
                                           verbose = True,
                                           path = (representation + '_' + protocol + '_opengan_model' + str(trial) + '.pt'))
            # these next two lines are used to apply multi-class labels
            # to the fakes when training the discriminator
            unmonitored_label = torch.zeros(61)
            unmonitored_label[60] = 1.0
            print('Starting to train now for', representation, protocol)
            for epoch in range(240):
                lossD = 0.0
                lossG = 0.0
                train_mon_iter = iter(train_mon_loader)
                train_unmon_iter = iter(train_unmon_loader)
                x_batches = []
                y_batches = []
                for _ in range(len(train_mon_loader)):
                    # get next batches and unpack them
                    train_mon = next(train_mon_iter)
                    x_train_mon, y_train_mon = train_mon
                    x_train_mon_features = model.extract_features_flattened(x_train_mon.to(device))
                    # add a 61st class value of 0.0, because the
                    # labels were just 0-59 for the monitored-only task
                    labels_60 = torch.zeros(len(y_train_mon), 1)
                    y_train_mon = torch.cat((y_train_mon, labels_60), dim=1)
                    
                    train_unmon = next(train_unmon_iter)
                    x_train_unmon, y_train_unmon = train_unmon
                    x_train_unmon_features = model.extract_features_flattened(x_train_unmon.to(device))
                    
                    # set the number of fake monitored instances to the
                    # balance of the number of real unmonitored instances
                    # while maintaining a ratio of 1:2 real monitored to
                    # real unmonitored + fake monitored
                    noise = torch.randn(int(lambda_g * 0.67 * BEST_HYPERPARAMETERS[representation + '_' + protocol]['batch_size']),
                                        100, device=device)
                    netG.eval()
                    fake_features = netG(noise).detach()
                    # label all fakes as unmonitored
                    y_fake_60 = unmonitored_label.repeat(len(fake_features), 1)
                    
                    # train discriminator on real monitored, real unmonitored,
                    # and fake monitored all at once so that batch normalization
                    # statistics are calculated the same way that the baseline
                    # model calculates them when lambda_g = 0
                    optimizerD.zero_grad()
                    netD.train()
                    if lambda_g < 1.0:
                        features = torch.cat([x_train_mon_features, x_train_unmon_features, fake_features])
                        y_train = torch.cat([y_train_mon, y_train_unmon, y_fake_60])
                    # don't include that one instance from the train_unmon_loader
                    # that we drew as a workaround to prevent an error
                    else:
                        features = torch.cat([x_train_mon_features, fake_features])
                        y_train = torch.cat([y_train_mon, y_fake_60])
                    # only update the discriminator every few epochs
                    if epoch % lambda_e == 0:
                        output = netD(features, training = True)
                        loss = criterionD(output, y_train.to(device))
                        loss.backward()
                        optimizerD.step()
                        lossD += loss.item()
                    
                    # every few epochs, produce a t-SNE to show the relationship
                    # between the monitored, unmonitored, and fake instances...
                    # start here by saving the instances from each batch for later
                    if epoch % 50 == 0 and plot_tsne:
                        x_batches.append(features.detach().cpu())
                        # convert convert from one-hot encoding to single labels here too
                        y_batches.append(torch.full((y_train_mon.shape[0],), 0, dtype=torch.float32))
                        y_batches.append(torch.full((y_train_unmon.shape[0],), 1, dtype=torch.float32))
                        # re-label the fakes to 61 so we can distinguish between them and the
                        # real unmonitored instances on the t-SNE plot
                        y_batches.append(torch.full((y_fake_60.shape[0],), 2, dtype=torch.float32))
                        
                    # train generator with the discriminator's binary
                    # predictions on fake monitored instances
                    optimizerG.zero_grad()
                    netD.eval()
                    netG.train()
                    noise = torch.randn(len(y_train), 100, device=device)
                    features = netG(noise)
                    output = netD(features, training = False)
                    # convert logits to predicted probabilities
                    preds = torch.softmax(output, dim=1)
                    # find the MSP for monitored classes
                    preds_binary, _ = torch.max(preds[:, :60], dim=1)
                    # label all fakes as monitored
                    y_fake_ones = torch.ones(len(y_train), dtype=torch.float32)
                    # using BCELoss now because we're agnostic about what monitored
                    # classes for which the generator outputs fakes
                    loss = criterionG(preds_binary, y_fake_ones.to(device))
                    x_train_mon_features = model.extract_features_flattened(x_train_mon.to(device))
                    feature_loss = torch.norm(x_train_mon_features.mean(dim=0) - features.mean(dim=0), p=2)
                    combined_loss = loss + (lambda_fm * feature_loss)
                    combined_loss.backward()
                    optimizerG.step()
                    lossG += combined_loss.item()

                # check discriminator's performance on fakes and 
                # the validation set
                val_loss = 0.0
                with torch.no_grad():
                    netD.eval()
                    netG.eval()
                    noise = torch.randn(50, 100, device=device)
                    fakes = netG(noise)
                    output = netD(fakes, training = False)
                    preds = torch.softmax(output, dim=1)
                    preds_binary, _ = torch.max(preds[:, :60], dim=1)
                    mean_pred_fake = preds_binary.detach().cpu().mean()
                
                    for x_val, y_val in val_loader:
                        netD.eval()
                        features_val = model.extract_features_flattened(x_val.to(device))
                        logits_val = netD(features_val, training = False)
                        loss = criterionD(logits_val, y_val.to(device))
                        val_loss += loss.item()
                        
                        # compute PR-AUC over the validation set
                        y_val_binary = 1 - y_val[:, 60]
                        preds_val = torch.softmax(logits_val, dim=1)
                        preds_val_binary, _ = torch.max(preds_val[:, :60], dim=1)
                        precisions, recalls, thresholds = precision_recall_curve(y_val_binary.cpu(), preds_val_binary.cpu())
                        pr_auc = auc(recalls, precisions)

                # every few epochs, produce a t-SNE to show the relationship
                # between the monitored, unmonitored, and fake instances...
                # we only produce a plot from the epoch before the discriminator
                # updates again, when it is most often fooled by the generator
                if epoch % 50 == 0 and plot_tsne:
                    print('tsne: concatenating batches')
                    x_batches_np = torch.cat(x_batches).numpy()
                    y_batches_np = torch.cat(y_batches).numpy()
                    print('tsne: reducing dimensionality')
                    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200)
                    tsne_results = tsne.fit_transform(x_batches_np)
                    plt.figure(figsize=(16, 12))
                    unique_labels = numpy.unique(y_batches_np)  # Should be [0, 1, 2]
                    colors = ['blue', 'black', 'green']  # Colors for monitored, unmonitored, and fake
                    labels_dict = {0: 'Monitored', 1: 'Unmonitored', 2: 'Fake'}  # Labels for the legend
                    # Plot each category with its own color and label
                    for i, label in enumerate(unique_labels):
                        idxs = numpy.where(y_batches_np == label)
                        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], c=colors[i], label=labels_dict[label], alpha=0.5)
                    plt.title(f't-SNE Visualization of Training Data at Epoch {epoch}')
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.savefig(f'tsne_opengan_fea_dfnet_{protocol}_epoch_{epoch}.png')
                    plt.close()

                print(f'Epoch {epoch+1} \t lossD: {lossD} \t lossG: {lossG} \t mean_pred_fake: {mean_pred_fake} \t Val Loss: {val_loss / len(val_dataset)} \t Val PR-AUC: {pr_auc}')
                # check if this is a new low validation loss and, if so, save the discriminator
                #
                # otherwise increment the counter towards the patience limit
                early_stopping(-pr_auc, netD)
                if early_stopping.early_stop:
                    # we've reached the patience limit
                    print('Early stopping')
                    break
