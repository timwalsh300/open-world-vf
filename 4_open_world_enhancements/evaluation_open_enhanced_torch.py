# This takes no arguments on the command line. It
# loads the trained model(s), dataset splits,
# and does the evaluation.
#
# Outputs are the open-world performance as a precision-
# recall curve for the 64k HTTPS-only test set, and one
# for the 64k Tor test set, and the PR-AUC for each
# approach on each test set.

import torch
import mymodels_torch
import numpy
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt


INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}
                       
pr_curve_data = {}

def get_scores(test_loader, protocol, representation, approach):
    if approach == 'baseline':
        # we manually copy and paste these hyperparameters from the output of search_open.py
        BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                                'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model.pt'))
        model.to(device)
        model.eval()
        logits_batches = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                logits_batches.append(model(x_test.to(device)).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
            # Find the max softmax probability for any monitored
            # class. This will be fairly low if the argmax was 60 or
            # if the model was torn between two monitored classes.
            # We're implying the the probability of 60 is 1.0 - this.
            scores.append(max(preds[i][:60]))
        return scores

    # this is mostly copied from the baseline approach above...
    elif approach == 'temp_scaling':
        # we manually copy and paste these hyperparameters from the output of search_open.py
        BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                                'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model.pt'))
        model.to(device)
        model.eval()
        temp_model = mymodels_torch.TemperatureScaling()
        temp_model.load_state_dict(torch.load(representation + '_' + protocol + '_temp_scaling_model.pt'))
        temp_model.to(device)
        temp_model.eval()
        logits_batches = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                logits = model(x_test.to(device))
                logits_batches.append(temp_model(logits).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
            # Find the max softmax probability for any monitored
            # class. This will be fairly low if the argmax was 60 or
            # if the model was torn between two monitored classes.
            # We're implying the the probability of 60 is 1.0 - this.
            scores.append(max(preds[i][:60]))
        return scores
    
    elif approach == 'map':
        # we manually copy and paste these hyperparameters from the output of search_open.py
        BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                                'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_map_model.pt'))
        model.to(device)
        model.eval()
        logits_batches = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                logits_batches.append(model(x_test.to(device)).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
            # Find the max softmax probability for any monitored
            # class. This will be fairly low if the argmax was 60 or
            # if the model was torn between two monitored classes.
            # We're implying the the probability of 60 is 1.0 - this.
            scores.append(max(preds[i][:60]))
        return scores
    
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for protocol in ['https', 'tor']:
    for representation in ['dschuster16', 'schuster8']:
        try:
            test_tensors = torch.load(representation + '_' + protocol + '_test_tensors.pt')
            test_dataset = torch.utils.data.TensorDataset(*test_tensors)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=128,
                                                      shuffle=False)
            y_test_batches = []
            for _, y_test in test_loader:
                y_test_batches.append(y_test)
            y_test_concatenated = torch.cat(y_test_batches, dim = 0)
            y_test_np = y_test_concatenated.numpy()
            true_labels = numpy.argmax(y_test_np, axis = 1)
            true_binary = (true_labels < 60)
        except Exception as e:
            print(e)
            continue
            
        # I'll add more approaches to this list as I build them
        for approach in ['baseline', 'temp_scaling', 'map']:
            print('Getting scores for', protocol, approach)
            scores = get_scores(test_loader, protocol, representation, approach)
            precisions, recalls, thresholds = precision_recall_curve(true_binary, scores)
            # add these to a global data structure to plot later
            pr_curve_data[approach] = (precisions, recalls, thresholds)
            pr_auc = auc(recalls, precisions)
            print('PR-AUC:', pr_auc)

        # create and save the P-R curve figure
        colors = plt.cm.viridis(numpy.linspace(0, 1, len(pr_curve_data)))
        plt.figure(figsize=(16, 12))
        for (label, (precisions, recalls, _)), color in zip(pr_curve_data.items(), colors):
            plt.plot(recalls, precisions, label=label, color=color)
        plt.xlabel('Recall', fontsize = 32)
        plt.ylabel('Precision', fontsize = 32)
        protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
        plt.title('Precision-Recall Curve (' + protocol_string + ' 64k Test Set)', fontsize = 32)
        plt.legend(loc = 'lower left', fontsize = 20, title = 'Approach', title_fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.xlim(0.5, 1)
        plt.ylim(0.5, 1)
        plt.grid(True)
        plt.savefig('enhanced_pr_curve_' + protocol + '.png', dpi=300)
        pr_curve_data = {}
