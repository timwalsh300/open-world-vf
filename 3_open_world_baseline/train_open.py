import mymodels
from tensorflow import keras
import pickle
import numpy

# This takes no arguments on the command line. It just 
# loads the right model, hyperparameters, dataset splits, 
# and does the training.
#
# output is the trained model(s)

SHAPES = {'sirinam_wf': (5000, 1),
          'sirinam_vf': (25000, 1),
          'rahman': (25000, 1),
          'hayden': (25000, 1),
          'schuster2': (480, 1),
          'schuster4': (960, 1),
          'schuster8': (1920, 1),
          'dschuster8': (1920, 2),
          'schuster16': (3840, 1),
          'dschuster16': (3840, 2)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                        'dschuster16_tor': {'filters': 256, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 0.0005588278335759993, 'batch_size': 64},
                        'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}
                       }

def create_model(representation, protocol):
    if representation == 'sirinam_wf':
        model = mymodels.DFNet.build(SHAPES[representation], 61)
        lr = 0.002
    else:
        model = mymodels.DFNetTunable.build(SHAPES[representation], 61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        lr = BEST_HYPERPARAMETERS[representation + '_' + protocol]['lr']
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = lr),
                  loss = 'categorical_crossentropy',
                  metrics = [keras.metrics.Precision(class_id = 60),
                             keras.metrics.Recall(class_id = 60)])
    return model

def train_model(representation, protocol):
    with open(representation + '_open_world_' + protocol + '_splits.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    model = create_model(representation, protocol)
    # val_loss is a more common monitor and suggested by ChatGPT for imbalanced datasets but we can also
    # try val_accuracy because val_loss could result in finding a sharp anomaly in the loss landscape
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience = 10,
                                                   verbose = 1,
                                                   mode = 'min',
                                                   restore_best_weights = True)
    batch_size = 128 if representation == 'sirinam_wf' else BEST_HYPERPARAMETERS[representation + '_' + protocol]['batch_size']
    history = model.fit(splits['x_train'],
                        splits['y_train'],
                        epochs = 120,
                        callbacks = [early_stopping],
                        batch_size = batch_size,
                        validation_data = (splits['x_val'], splits['y_val']),
                        verbose = 2)
    return model

#for representation in ['dschuster16', 'schuster16', 'dschuster8', 'schuster8', 'schuster4', 'schuster2', 'hayden', 'rahman', 'sirinam_vf', 'sirinam_wf']:
for representation in ['dschuster16', 'schuster8']:
    for protocol in ['https', 'tor']:
        print('starting to train now for', representation, protocol)
        try:
            model = train_model(representation, protocol)
            model.save(representation + '_open_world_' + protocol + '_model.h5')
        except Exception as e:
            print(e)
