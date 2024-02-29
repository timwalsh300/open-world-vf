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
    # this is the MLE model that uses Max Softmax Probability or
    # Softmax Thresholding for OSR
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
                logits_batches.append(model(x_test.to(device), training = False).to('cpu'))
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

    # this is mostly copied from the baseline approach but adds
    # some lines to introduce the trained temp scaling layer produced by
    # train_temp_scaling_torch.py
    elif approach == 'temp_scaling':
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
                logits = model(x_test.to(device), training = False)
                logits_batches.append(temp_model(logits).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
            scores.append(max(preds[i][:60]))
        return scores
        
    # this is mostly copied from the baseline approach but loads
    # the trained MAP model with L2 prior regularization instead of the
    # baseline MLE model
    elif approach == 'map':
        BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                                'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        # train_map_torch.py does a hyperparameter search over a range of L2 coefficients and
        # saves the best model, at its lowest validation loss, at the following path
        model.load_state_dict(torch.load(representation + '_' + protocol + '_map_model.pt'))
        model.to(device)
        model.eval()
        logits_batches = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                logits_batches.append(model(x_test.to(device), training = False).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
            scores.append(max(preds[i][:60]))
        return scores
        
    # this is mostly copied from the baseline approach but loads
    # the trained MAP model with L2 prior regularization instead of the
    # baseline MLE model, and then does Monte Carlo Dropout and Bayesian
    # model averaging before using Max Softmax Probability or
    # Softmax Thresholding for OSR
    elif approach == 'mcd_msp':
        BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                                'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_map_model.pt'))
        model.to(device)
        model.eval()
        mc_samples = 50
        preds = numpy.zeros([mc_samples, 64600, 61])
        for i in range(mc_samples):
            logits_batches = []
            for x_test, y_test in test_loader:
                with torch.no_grad():
                    logits_batches.append(model(x_test.to(device), training = True).to('cpu'))
            logits_concatenated = torch.cat(logits_batches, dim = 0)
            preds[i] = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        preds_avg = preds.mean(axis = 0)
        scores = []
        for i in range(len(preds_avg)):
            scores.append(max(preds_avg[i][:60]))
        return scores
        
    # this loads a trained MAP model with L2 prior regularization and then does Monte
    # Carlo Dropout before using epistemic uncertainty to rank predictions when they
    # are for any monitored class
    # 
    # the intuition is that the epistemic uncertainty should be higher when we have
    # a false positive than when we have a true positive
    #
    # ranking both monitored and unmonitored predictions by epistemic uncertainty is
    # problematic because uncertainty will be low on both ends, so low epistemic
    # uncertainty doesn't mean that an instance is obviously monitored; it could be
    # obviously unmonitored too
    elif approach == 'mcd_uncertainty':
        BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                                'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_map_model.pt'))
        model.to(device)
        model.eval()
        mc_samples = 50
        preds = numpy.zeros([mc_samples, 64600, 61])
        entropies = numpy.zeros([mc_samples, 64600])
        for i in range(mc_samples):
            logits_batches = []
            for x_test, y_test in test_loader:
                with torch.no_grad():
                    logits_batches.append(model(x_test.to(device), training = True).to('cpu'))
            logits_concatenated = torch.cat(logits_batches, dim = 0)
            preds[i] = torch.softmax(logits_concatenated, dim=1).detach().numpy()
            entropies[i] = -1 * numpy.sum(preds[i] * numpy.log(preds[i] + 1e-9), axis = 1)
        preds_avg = preds.mean(axis = 0)
        # compute the total uncertainty for each test instance
        total_uncertainties = -1 * numpy.sum(preds_avg * numpy.log(preds_avg), axis = 1)
        # compute aleatoric uncertainty for each test instance
        aleatoric_uncertainties = numpy.mean(entropies, axis = 0)
        # compute the epistemic uncertainty for each test instance
        epistemic_uncertainties = total_uncertainties - aleatoric_uncertainties
        # we negate the epistemic uncertainties because precision_recall_curve() expects
        # a higher score to be associated with a higher confidence prediction for the
        # positive class, but a high epistemic uncertainty is a low confidence prediction
        scores = [-eu for eu in epistemic_uncertainties]
        # now the min of scores is the highest uncertainty
        highest_eu = min(scores)
        for i in range(64600):
            # if the prediction was unmonitored, we rank it as low as possible without
            # even considering the uncertainty
            if numpy.argmax(preds_avg[i]) == 60:
                scores[i] = highest_eu
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
        for approach in ['baseline', 'temp_scaling', 'map', 'mcd_msp', 'mcd_uncertainty']:
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
