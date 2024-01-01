import mymodels
from tensorflow import keras
import pickle
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# This takes no arguments on the command line. It just
# does the evaluation of the Schuster representation against
# Vimeo, over both HTTPS-only and Tor, in the closed-world 
# scenario and produces	some insights about the	model's	errors

BASE_PATH = '/home/timothy.walsh/VF/1_csv_to_pkl/'
DATA_POINTS = {'sirinam': 5000, 'hayden': 25000, 'schuster': 960}
BEST_HYPERPARAMETERS = {'schuster_https_youtube': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 2, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 1.864628268135616e-05, 'batch_size': 32},
                        'schuster_https_facebook': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.1, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 1.6011606405643378e-05, 'batch_size': 32},
                        'schuster_https_vimeo': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 5.135210339286033e-05, 'batch_size': 128},
                        'schuster_https_rumble': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.1, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 0.00021151668452050297, 'batch_size': 32},
                        'schuster_tor_youtube': {'filters': 32, 'kernel': 16, 'conv_stride': 1, 'pool': 2, 'pool_stride': 2, 'conv_dropout': 0.1, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.000718813082694037, 'batch_size': 32},
                        'schuster_tor_facebook': {'filters': 64, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 16, 'conv_dropout': 0.2, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.00026361099810913094, 'batch_size': 32},
                        'schuster_tor_vimeo': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 4, 'pool_stride': 4, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.0019529496617439444, 'batch_size': 64},
                        'schuster_tor_rumble': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.00045252193030865536, 'batch_size': 128},
                        'hayden_https_youtube': {'filters': 32, 'kernel': 32, 'conv_stride': 2, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.4, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 0.0005833537224684166, 'batch_size': 32},
                        'hayden_https_facebook': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 16, 'conv_dropout': 0.1, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.00020025077461198002, 'batch_size': 32},
                        'hayden_https_vimeo': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.4, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.0011015437229458696, 'batch_size': 32},
                        'hayden_https_rumble': {'filters': 128, 'kernel': 16, 'conv_stride': 2, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.000689611966667205, 'batch_size': 64},
                        'hayden_tor_youtube': {'filters': 64, 'kernel': 16, 'conv_stride': 1, 'pool': 2, 'pool_stride': 4, 'conv_dropout': 0.6, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.002922704350075904, 'batch_size': 128},
                        'hayden_tor_facebook': {'filters': 256, 'kernel': 8, 'conv_stride': 2, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.4, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.011540629715338448, 'batch_size': 128},
                        'hayden_tor_vimeo': {'filters': 256, 'kernel': 8, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.00147750289758336, 'batch_size': 64},
                        'hayden_tor_rumble': {'filters': 128, 'kernel': 8, 'conv_stride': 2, 'pool': 4, 'pool_stride': 2, 'conv_dropout': 0.6, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.000963159520134116, 'batch_size': 32}
                        }

def create_model(representation, protocol, platform):
    if representation == 'sirinam':
        model = mymodels.DFNet.build((DATA_POINTS[representation], 1), 60)
        lr = 0.002
    else:
        model = mymodels.DFNetTunable.build((DATA_POINTS[representation], 1), 60,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform])
        lr = BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform]['lr']
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

def train_model(representation, protocol, platform):
    with open(BASE_PATH + representation + '_monitored_' + protocol + '_' + platform + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    model = create_model(representation, protocol, platform)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience = 10,
                                                   verbose = 1,
                                                   mode = 'min',
                                                   restore_best_weights = True)
    batch_size = 128 if representation == 'sirinam' else BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform]['batch_size']
    history = model.fit(splits['x_train'],
                        splits['y_train'],
                        epochs = 120,
                        callbacks = [early_stopping],
                        batch_size = batch_size,
                        validation_data = (splits['x_val'], splits['y_val']),
                        verbose = 2)
    return model

def analyze_errors(model, representation, protocol, platform):
    with open(BASE_PATH + representation + '_monitored_' + protocol + '_' + platform + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    print(representation, protocol, platform)
    probs = model.predict(splits['x_test'])
    predicted_classes = np.argmax(probs, axis=1)
    true_classes = np.argmax(splits['y_test'], axis=1)
    print(classification_report(true_classes, predicted_classes))
    misclassified_indices = np.where(predicted_classes != true_classes)[0]
    misclassified_probs = probs[misclassified_indices]
    max_probs_misclassified = np.max(misclassified_probs, axis=1)
    plt.hist(max_probs_misclassified, bins=50, cumulative=True, density=True, histtype='step', alpha=0.8, color='k')
    plt.xlabel('Max Predicted Probability')
    plt.ylabel('Cumulative Frequency')
    plt.title('CDF of Maximum Predicted Probabilities for Misclassified Instances')
    plt.grid(True)
    plt.savefig(BASE_PATH + representation + '_' + protocol + '_' + platform + '_misclassified_cdf_plot.png',
                dpi = 300,
                bbox_inches = 'tight')
    plt.close()

for protocol in ['https', 'tor']:
    model = train_model('schuster', protocol, 'vimeo')
    analyze_errors(model, 'schuster', protocol, 'vimeo')
