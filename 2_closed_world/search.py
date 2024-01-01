import mymodels
from ray import tune, train
from ray.tune.schedulers import AsyncHyperBandScheduler
from tensorflow import keras
import numpy
import pickle
import sys

# This takes three arguments on the command line to load the
# desired dataset splits...
#
# 1: sirinam_wf, sirinam_vf, rahman, hayden, or schuster
# 2: tor or https
# 3: youtube, facebook, vimeo, or rumble
#
# output is the best hyperparameters

BASE_PATH = '/home/timothy.walsh/VF/1_csv_to_pkl/'
REPRESENTATION = sys.argv[1]
PROTOCOL = sys.argv[2]
PLATFORM = sys.argv[3]
SHAPES = {'sirinam_vf': (25000, 1),
          'rahman': (25000, 1),
          'hayden': (25000, 1),
          'schuster2': (480, 1),
          'schuster4': (960, 1),
          'schuster8': (1920, 1),
          'dschuster8': (1920, 2),
          'sschuster8': (1920, 1),
          'rschuster8': (1920, 1),
          'schuster16': (3840, 1),
          'dschuster16': (3840, 2),
          'sschuster16': (3840, 1),
          'rschuster16': (3840, 1)}

def create_model(hyperparameters):
    model = mymodels.DFNetTunable.build(SHAPES[REPRESENTATION], 60, hyperparameters)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hyperparameters['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(config):
    # if we're only going to work on bytes sent or received, we'll load the dschuster splits
    if 'sschuster' in REPRESENTATION or 'rschuster' in REPRESENTATION:
        mod_rep = 'd' + REPRESENTATION[1:]
    else:
        mod_rep = REPRESENTATION
    with open(BASE_PATH + mod_rep + '_monitored_' + PROTOCOL + '_' + PLATFORM + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    # now select only the channel that we want from the dschuster x splits
    if 'sschuster' in REPRESENTATION:
        x_train = splits['x_train'][:, :, 0]
        x_train = numpy.expand_dims(x_train, axis=-1)
        x_val = splits['x_val'][:, :, 0]
        x_val = numpy.expand_dims(x_val, axis=-1)
    if 'rschuster' in REPRESENTATION:
        x_train = splits['x_train'][:, :, 1]
        x_train = numpy.expand_dims(x_train, axis=-1)
        x_val = splits['x_val'][:, :, 1]
        x_val = numpy.expand_dims(x_val, axis=-1)
    else:
        x_train = splits['x_train']
        x_val = splits['x_val']
    model = create_model(config)
    history = model.fit(x_train,
                        splits['y_train'],
                        epochs = 30,
                        batch_size = config['batch_size'],
                        validation_data = (x_val, splits['y_val']))
    val_loss = history.history['val_loss'][-1]
    return {'val_loss': val_loss}

def tune_model():
    sched = AsyncHyperBandScheduler(max_t = 30)
    tuner = tune.Tuner(tune.with_resources(train_model, resources = {'cpu': 2, 'gpu': 1}),
                       tune_config = tune.TuneConfig(metric = 'val_loss',
                                                     mode = 'min',
                                                     scheduler = sched,
                                                     num_samples = 80),
                       run_config = train.RunConfig(),
                       param_space = {'filters': tune.choice([32, 64, 128, 256]),
                                      'kernel': tune.choice([4, 8, 16, 32]),
                                      'conv_stride': tune.choice([1, 2, 4, 8, 16]),
                                      'pool': tune.choice([2, 4, 8, 16]),
                                      'pool_stride': tune.choice([1, 2, 4, 8, 16]),
                                      'conv_dropout': tune.choice([0.1, 0.2, 0.4, 0.6, 0.8]),
                                      'fc_neurons': tune.choice([128, 256, 512, 1024]),
                                      'fc_init': tune.choice(['glorot_uniform', 'he_normal']),
                                      'fc_activation': tune.choice(['relu', 'elu']),
                                      'fc_dropout': tune.choice([0.1, 0.2, 0.4, 0.6, 0.8]),
                                      'lr': tune.loguniform(1e-5, 1e-1),
                                      'batch_size': tune.choice([32, 64, 128])})
    results = tuner.fit()
    print('Best hyperparameters found were:', results.get_best_result().config)

tune_model()
