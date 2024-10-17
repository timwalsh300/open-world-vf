# This takes no arguments on the command line
#
# Outputs are the trained models.

import torch
import mymodels_torch
import numpy
import statistics
import pickle

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster8': (2, 1920),
                'schuster16': (1, 3840),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the closed-world searches
BEST_HYPERPARAMETERS = {'dschuster16_https_youtube': {'filters': 128, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 0.00014585969525380388, 'batch_size': 32},
                        'dschuster16_https_facebook': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 4, 'pool_stride': 16, 'conv_dropout': 0.1, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.00017903551714471978, 'batch_size': 32},
                        'dschuster16_https_vimeo': {'filters': 256, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 6.0487458111998665e-05, 'batch_size': 32},
                        'dschuster16_https_rumble': {'filters': 64, 'kernel': 32, 'conv_stride': 2, 'pool': 4, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.00010667937830812648, 'batch_size': 128},
                        'dschuster16_tor_youtube': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.00048775093030182354, 'batch_size': 64},
                        'dschuster16_tor_facebook': {'filters': 256, 'kernel': 16, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 2.01341322605435e-05, 'batch_size': 128},
                        'dschuster16_tor_vimeo': {'filters': 128, 'kernel': 16, 'conv_stride': 1, 'pool': 4, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 0.00032731760817436225, 'batch_size': 64},
                        'dschuster16_tor_rumble': {'filters': 128, 'kernel': 8, 'conv_stride': 1, 'pool': 4, 'pool_stride': 4, 'conv_dropout': 0.4, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.005825171851117097, 'batch_size': 128},

                        'schuster16_https_youtube': {'filters': 64, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.00023804257990210477, 'batch_size': 64},
                        'schuster16_https_facebook': {'filters': 256, 'kernel': 32, 'conv_stride': 2, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.1, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.0016171048352516617, 'batch_size': 64},
                        'schuster16_https_vimeo': {'filters': 32, 'kernel': 16, 'conv_stride': 2, 'pool': 2, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0002051723906560109, 'batch_size': 64},
                        'schuster16_https_rumble': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.001803112876892599, 'batch_size': 32},
                        'schuster16_tor_youtube': {'filters': 64, 'kernel': 16, 'conv_stride': 2, 'pool': 4, 'pool_stride': 2, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.0003772328299447977, 'batch_size': 64},
                        'schuster16_tor_facebook': {'filters': 256, 'kernel': 32, 'conv_stride': 2, 'pool': 16, 'pool_stride': 1, 'conv_dropout': 0.8, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0007326039695447873, 'batch_size': 32},
                        'schuster16_tor_vimeo': {'filters': 128, 'kernel': 32, 'conv_stride': 2, 'pool': 4, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.00013460297668644165, 'batch_size': 64},
                        'schuster16_tor_rumble': {'filters': 128, 'kernel': 16, 'conv_stride': 2, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.0028809197407786325, 'batch_size': 128},
                        
                        'dschuster8_https_youtube': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0003728796604381587, 'batch_size': 32},
                        'dschuster8_https_facebook': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.6, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.00048124756418014127, 'batch_size': 128},
                        'dschuster8_https_vimeo': {'filters': 128, 'kernel': 4, 'conv_stride': 1, 'pool': 2, 'pool_stride': 2, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.8, 'lr': 0.00024224291110144496, 'batch_size': 64},
                        'dschuster8_https_rumble': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 2, 'pool_stride': 2, 'conv_dropout': 0.8, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 0.00014004061424856013, 'batch_size': 64},
                        'dschuster8_tor_youtube': {'filters': 256, 'kernel': 8, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 0.0002976812130068175, 'batch_size': 64},
                        'dschuster8_tor_facebook': {'filters': 64, 'kernel': 8, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.00012472037149085724, 'batch_size': 32},
                        'dschuster8_tor_vimeo': {'filters': 32, 'kernel': 32, 'conv_stride': 1, 'pool': 2, 'pool_stride': 2, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0026518717644831396, 'batch_size': 32},
                        'dschuster8_tor_rumble': {'filters': 64, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.003092583631892268, 'batch_size': 32},
                        
                        'schuster8_https_youtube': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.4, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.00033127762445277437, 'batch_size': 128},
                        'schuster8_https_facebook': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.00021330635803808715, 'batch_size': 32},
                        'schuster8_https_vimeo': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 4, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 4.2070406695252937e-05, 'batch_size': 128},
                        'schuster8_https_rumble': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.6, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.0005615133920187598, 'batch_size': 64},
                        'schuster8_tor_youtube': {'filters': 128, 'kernel': 8, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.6, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.0023199540882264144, 'batch_size': 128},
                        'schuster8_tor_facebook': {'filters': 64, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.004981052944639142, 'batch_size': 64},
                        'schuster8_tor_vimeo': {'filters': 256, 'kernel': 8, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.00014940538775594852, 'batch_size': 128},
                        'schuster8_tor_rumble': {'filters': 64, 'kernel': 32, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.0009612315385594381, 'batch_size': 64}}

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for protocol in ['https', 'tor']:
  for platform in ['youtube', 'facebook', 'vimeo', 'rumble']:
    results = {}
    for training_region in ['africa', 'brazil', 'frankfurt', 'london', 'oregon', 'seoul', 'stockholm', 'sydney', 'uae', 'virginia']:
      for testing_region in ['africa', 'brazil', 'frankfurt', 'london', 'oregon', 'seoul', 'stockholm', 'sydney', 'uae', 'virginia']:
        for representation in ['schuster8', 'dschuster8', 'schuster16', 'dschuster16']:
          try:
            # if they exist, load the data tensors that resulted from raw_to_csv.py and csv_to_pt.py
            test_tensors = torch.load('data_splits/test_' + representation + '_' + protocol + '_' + platform + '_' + testing_region + '.pt')
            test_dataset = torch.utils.data.TensorDataset(*test_tensors)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size = BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform]['batch_size'],
                                                      shuffle=False)
          except Exception as e:
            # we expect to hit this condition for many irrelevant combinations
            print(e)
            continue
          trial_accuracies = []
          for trial in range(10):
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation], 60,
                                                BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform])
            model.load_state_dict(torch.load('models/' + representation + '_' + protocol + '_' + platform + '_' + training_region + '_model'  + str(trial) + '.pt'))
            model.to(device)
            model.eval()
            print('Testing', representation, protocol, platform, training_region, testing_region, trial)
            correct_predictions = 0
            total_predictions = 0
            model.eval()
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    output = model(x_test.to(device), training = False)
                    y_test_indices = torch.argmax(y_test, dim=1)
                    predicted_classes = torch.argmax(output, dim=1)
                    correct_predictions += (predicted_classes == y_test_indices.to(device)).sum().item()
                    total_predictions += y_test_indices.size(0)
            trial_accuracies.append(correct_predictions / total_predictions)
          print('mean and stddev (accuracy) ', statistics.mean(trial_accuracies), statistics.stdev(trial_accuracies))
          results[training_region + '_' + testing_region] = (statistics.mean(trial_accuracies), statistics.stdev(trial_accuracies))
    with open('closed_' + protocol + '_' + platform + '_results.pkl', 'wb') as handle:
        pickle.dump(results, handle)        
            

