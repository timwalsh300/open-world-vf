# This takes no arguments on the command line
#
# Outputs the best values of T and trained
# temp scaling layers.

import torch
import mymodels_torch
import numpy
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                        'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

# this plots and saves a reliability diagram and then
# returns the calculated expected calibration error
def check_calibration(logits, y_val, approach, protocol):
    preds = torch.softmax(logits, dim=1).detach().numpy()
    scores = []
    for i in range(len(preds)):
        # Find the max softmax probability for any monitored
        # class. This will be fairly low if the argmax was 60 or
        # if the model was torn between two monitored classes.
        # We're implying the the probability of 60 is 1.0 - this.
        scores.append(max(preds[i][:60]))
    true_labels = numpy.argmax(y_val, axis = 1)
    true_binary = (true_labels < 60)
    prob_true, prob_pred = calibration_curve(true_binary,
                                             scores,
                                             n_bins = 10,
                                             strategy = 'uniform')
    
    # Plot and save a reliability diagram
    bin_edges = numpy.linspace(0, 1, 11)
    bin_width = numpy.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_width / 2
    bin_counts = numpy.histogram(scores, bins=bin_edges)[0]
    plt.figure(figsize=(16, 12))
    plt.bar(bin_centers, prob_true, width = bin_width, align = 'center', alpha = 0.5, edgecolor='b', label=approach)
    for i, count in enumerate(bin_counts):
        plt.text(bin_centers[i], prob_true[i], f' {count}', verticalalignment= 'bottom', horizontalalignment = 'center')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability Bins and Number of Instances', fontsize = 32)
    plt.ylabel('True Positive Frequency', fontsize = 32)
    protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'    
    plt.title('Reliability Diagram (' + protocol_string + ' Val Set)', fontsize = 32)
    plt.legend(fontsize = 20)
    plt.savefig('reliability_' + protocol + '_' + approach + '.png', dpi=300)
    
    # Calculate the absolute difference between prob_true and prob_pred
    abs_diff = numpy.abs(prob_true - prob_pred)
    # Calculate the weights for each bin as the proportion of samples in each bin
    weights = bin_counts / numpy.sum(bin_counts)
    # Calculate the ECE as the weighted average of the absolute differences
    return numpy.sum(weights * abs_diff)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for representation in ['dschuster16', 'schuster8']:
    for protocol in ['https', 'tor']:
        try:
                val_tensors = torch.load(representation + '_' + protocol + '_val_tensors.pt')
                val_dataset = torch.utils.data.TensorDataset(*val_tensors)
                val_loader = torch.utils.data.DataLoader(val_dataset,
                                                         batch_size=len(val_dataset),
                                                         shuffle=False)
                model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                    61,
                                                    BEST_HYPERPARAMETERS[representation + '_' + protocol])
                model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model0.pt'))
                model.to(device)
                model.eval()
        except Exception as e:
            print(e)
            continue

        criterion = torch.nn.CrossEntropyLoss()
        temp_model = mymodels_torch.TemperatureScaling()
        temp_model.to(device)
        optimizer = torch.optim.LBFGS(temp_model.parameters(), lr=0.01, max_iter=100)

        print('Getting unscaled predicted logits for', representation, protocol)
        # this loops only once
        for x_val, y_val in val_loader:
            with torch.no_grad():
                logits = model(x_val.to(device), training = False)
            print('Baseline model ECE:', check_calibration(logits.to('cpu'), y_val, 'baseline', protocol))

#            print('Finding the best temperature T for', representation, protocol)
#            def eval():
#                optimizer.zero_grad()
#                loss = criterion(temp_model(logits), y_val.to(device))
#                loss.backward()
#                return loss
#            optimizer.step(eval)
        print('Optimal temperature: %.3f' % temp_model.temperature)
        print('Temperature Scaling model ECE:', check_calibration(temp_model(logits).to('cpu'), y_val.to('cpu'), 'temp_scaling', protocol))
        torch.save(temp_model.state_dict(), (representation + '_' + protocol + '_temp_scaling_model0.pt'))
