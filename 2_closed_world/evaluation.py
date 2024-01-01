import mymodels
from tensorflow import keras
import pickle
import numpy

# This takes no arguments on the command line. It just 
# iterates over all the combinations to load the right
# model, hyperparameters, dataset splits, and do the
# training and evaluation.
#
# output is the closed-world performance for each combination

BASE_PATH = '/home/timothy.walsh/VF/1_csv_to_pkl/'

SHAPES = {'sirinam_wf': (5000, 1),
          'sirinam_vf': (25000, 1),
          'rahman': (25000, 1),
          'hayden': (25000, 1),
          'schuster2': (480, 1),
          'schuster4': (960, 1),
          'schuster8': (1920, 1),
          'dschuster8': (1920, 2),
          'rschuster8': (1920, 1),
          'schuster16': (3840, 1),
          'dschuster16': (3840, 2)}
          
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
                        
                        'rschuster8_https_youtube': {'filters': 256, 'kernel': 32, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 1.4118765691175257e-05, 'batch_size': 128},
                        'rschuster8_https_facebook': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 4, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 2.417176223450153e-05, 'batch_size': 64},
                        'rschuster8_https_vimeo': {'filters': 256, 'kernel': 8, 'conv_stride': 2, 'pool': 2, 'pool_stride': 1, 'conv_dropout': 0.1, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 7.677517393863275e-05, 'batch_size': 32},
                        'rschuster8_https_rumble': {'filters': 256, 'kernel': 16, 'conv_stride': 2, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.0001220996142106429, 'batch_size': 128},
                        'rschuster8_tor_youtube': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.0002815868307774817, 'batch_size': 128},
                        'rschuster8_tor_facebook': {'filters': 128, 'kernel': 16, 'conv_stride': 2, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.2, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.00020112681305363498, 'batch_size': 64},
                        'rschuster8_tor_vimeo': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 4, 'pool_stride': 1, 'conv_dropout': 0.6, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0035440071409199333, 'batch_size': 128},
                        'rschuster8_tor_rumble': {'filters': 128, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.00014539713566821843, 'batch_size': 64},
                        
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
                        'schuster8_tor_rumble': {'filters': 64, 'kernel': 32, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.0009612315385594381, 'batch_size': 64},
                        
                        'schuster4_https_youtube': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 2, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.00011186808965293499, 'batch_size': 32},
                        'schuster4_https_facebook': {'filters': 64, 'kernel': 32, 'conv_stride': 1, 'pool': 2, 'pool_stride': 2, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.0003218102629526839, 'batch_size': 64},
                        'schuster4_https_vimeo': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 16, 'pool_stride': 1, 'conv_dropout': 0.8, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 7.112090427381767e-05, 'batch_size': 128},
                        'schuster4_https_rumble': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.8, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.00012254721175909973, 'batch_size': 32},
                        'schuster4_tor_youtube': {'filters': 64, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.0012670991299125376, 'batch_size': 64},
                        'schuster4_tor_facebook': {'filters': 256, 'kernel': 32, 'conv_stride': 1, 'pool': 2, 'pool_stride': 2, 'conv_dropout': 0.8, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 4.7682879334433124e-05, 'batch_size': 32},
                        'schuster4_tor_vimeo': {'filters': 128, 'kernel': 4, 'conv_stride': 1, 'pool': 4, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0004046366122562827, 'batch_size': 32},
                        'schuster4_tor_rumble': {'filters': 256, 'kernel': 16, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.00026078239913358305, 'batch_size': 64},
                        
                        'schuster2_https_youtube': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.4, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 2.9564954871409854e-05, 'batch_size': 32},
                        'schuster2_https_facebook': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.0006530405695908705, 'batch_size': 128},
                        'schuster2_https_vimeo': {'filters': 32, 'kernel': 32, 'conv_stride': 1, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.1, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 4.622050175634573e-05, 'batch_size': 64},
                        'schuster2_https_rumble': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.8, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.002194990137552231, 'batch_size': 32},
                        'schuster2_tor_youtube': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.6, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.0007853926408476579, 'batch_size': 32},
                        'schuster2_tor_facebook': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.6, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0001319574728116756, 'batch_size': 32},
                        'schuster2_tor_vimeo': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.002479615233199336, 'batch_size': 128},
                        'schuster2_tor_rumble': {'filters': 64, 'kernel': 32, 'conv_stride': 1, 'pool': 2, 'pool_stride': 2, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.0006748639113008366, 'batch_size': 64},
                        
                        'hayden_https_youtube': {'filters': 256, 'kernel': 16, 'conv_stride': 2, 'pool': 16, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.0008760313850537248, 'batch_size': 64},
                        'hayden_https_facebook': {'filters': 64, 'kernel': 32, 'conv_stride': 2, 'pool': 16, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.8, 'lr': 0.0003827939266081777, 'batch_size': 32},
                        'hayden_https_vimeo': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 0.007443031319414417, 'batch_size': 32},
                        'hayden_https_rumble': {'filters': 256, 'kernel': 32, 'conv_stride': 2, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.6, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.00011637970290420495, 'batch_size': 32},
                        'hayden_tor_youtube': {'filters': 128, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.6, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.0001974253716710011, 'batch_size': 64},
                        'hayden_tor_facebook': {'filters': 128, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 0.0008144031881207351, 'batch_size': 64},
                        'hayden_tor_vimeo': {'filters': 64, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.1, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.002367072317263736, 'batch_size': 32},
                        'hayden_tor_rumble': {'filters': 32, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.6, 'lr': 0.0004084796183637604, 'batch_size': 32},
                        
                        'rahman_https_youtube': {'filters': 256, 'kernel': 32, 'conv_stride': 2, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.6, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0008492931273537887, 'batch_size': 128},
                        'rahman_https_facebook': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 8, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.0023523339904593364, 'batch_size': 32},
                        'rahman_https_vimeo': {'filters': 256, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.4, 'fc_neurons': 256, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 2.8296418744970307e-05, 'batch_size': 64},
                        'rahman_https_rumble': {'filters': 32, 'kernel': 32, 'conv_stride': 2, 'pool': 4, 'pool_stride': 2, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 0.01722452367704241, 'batch_size': 64},
                        'rahman_tor_youtube': {'filters': 128, 'kernel': 16, 'conv_stride': 2, 'pool': 16, 'pool_stride': 1, 'conv_dropout': 0.2, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 9.53816853901861e-05, 'batch_size': 128},
                        'rahman_tor_facebook': {'filters': 64, 'kernel': 32, 'conv_stride': 2, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.00013184911194525696, 'batch_size': 128},
                        'rahman_tor_vimeo': {'filters': 128, 'kernel': 32, 'conv_stride': 2, 'pool': 16, 'pool_stride': 1, 'conv_dropout': 0.6, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.00030710853631361044, 'batch_size': 32},
                        'rahman_tor_rumble': {'filters': 128, 'kernel': 8, 'conv_stride': 2, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.00036182039849402305, 'batch_size': 64},
                        
                        'sirinam_vf_https_youtube': {'filters': 64, 'kernel': 32, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.6, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.8, 'lr': 0.0018356440283095682, 'batch_size': 32},
                        'sirinam_vf_https_facebook': {'filters': 128, 'kernel': 8, 'conv_stride': 2, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.2, 'lr': 0.01482212430327031, 'batch_size': 32},
                        'sirinam_vf_https_vimeo': {'filters': 128, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.4, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 0.009548144600801755, 'batch_size': 64},
                        'sirinam_vf_https_rumble': {'filters': 128, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.4, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 0.0035650992591760083, 'batch_size': 128},
                        'sirinam_vf_tor_youtube': {'filters': 64, 'kernel': 32, 'conv_stride': 2, 'pool': 2, 'pool_stride': 1, 'conv_dropout': 0.6, 'fc_neurons': 128, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.1, 'lr': 0.0027645738221079358, 'batch_size': 32},
                        'sirinam_vf_tor_facebook': {'filters': 128, 'kernel': 8, 'conv_stride': 2, 'pool': 8, 'pool_stride': 2, 'conv_dropout': 0.4, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.025299541583662054, 'batch_size': 128},
                        'sirinam_vf_tor_vimeo': {'filters': 128, 'kernel': 32, 'conv_stride': 1, 'pool': 16, 'pool_stride': 8, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 0.0001609000890252852, 'batch_size': 64},
                        'sirinam_vf_tor_rumble': {'filters': 64, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 16, 'conv_dropout': 0.2, 'fc_neurons': 512, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 0.0002230290756216252, 'batch_size': 64}
                        }

def create_model(representation, protocol, platform):
    if representation == 'sirinam_wf':
        model = mymodels.DFNet.build(SHAPES[representation], 60)
        lr = 0.002
    else:
        model = mymodels.DFNetTunable.build(SHAPES[representation], 60,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform])
        lr = BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform]['lr']
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

def train_model(representation, protocol, platform):
    # if we're only going to work on bytes sent or received, we'll load the dschuster splits
    if 'sschuster' in representation or 'rschuster' in representation:
        mod_rep = 'd' + representation[1:]
    else:
        mod_rep = representation
    with open(BASE_PATH + mod_rep + '_monitored_' + protocol + '_' + platform + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    # now select only the channel that we want from the dschuster x splits
    if 'sschuster' in representation:
        x_train = splits['x_train'][:, :, 0]
        x_train = numpy.expand_dims(x_train, axis=-1)
        x_val = splits['x_val'][:, :, 0]
        x_val = numpy.expand_dims(x_val, axis=-1)
    if 'rschuster' in representation:
        x_train = splits['x_train'][:, :, 1]
        x_train = numpy.expand_dims(x_train, axis=-1)
        x_val = splits['x_val'][:, :, 1]
        x_val = numpy.expand_dims(x_val, axis=-1)
    else:
        x_train = splits['x_train']
        x_val = splits['x_val']
    model = create_model(representation, protocol, platform)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience = 10,
                                                   verbose = 1,
                                                   mode = 'min',
                                                   restore_best_weights = True)
    batch_size = 128 if representation == 'sirinam_wf' else BEST_HYPERPARAMETERS[representation + '_' + protocol + '_' + platform]['batch_size']
    history = model.fit(x_train,
                        splits['y_train'],
                        epochs = 120,
                        callbacks = [early_stopping],
                        batch_size = batch_size,
                        validation_data = (x_val, splits['y_val']),
                        verbose = 2)
    return model

def evaluate_model(model, representation, protocol, platform, results):
    # if we're only going to work on bytes sent or received, we'll load the dschuster splits
    if 'sschuster' in representation or 'rschuster' in representation:
        mod_rep = 'd' + representation[1:]
    else:
        mod_rep = representation
    with open(BASE_PATH + mod_rep + '_monitored_' + protocol + '_' + platform + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    # now select only the channel that we want from the dschuster x splits
    if 'sschuster' in representation:
        x_test = splits['x_test'][:, :, 0]
        x_test = numpy.expand_dims(x_test, axis=-1)
    if 'rschuster' in representation:
        x_test = splits['x_test'][:, :, 1]
        x_test = numpy.expand_dims(x_test, axis=-1)
    else:
        x_test = splits['x_test']
    print(representation, protocol, platform)
    loss, accuracy = model.evaluate(x_test,
                                    splits['y_test'],
                                    verbose = 2)
    results[representation + '_' + protocol + '_' + platform] = accuracy

results = {}
#for representation in ['dschuster16', 'schuster16', 'dschuster8', 'schuster8', 'schuster4', 'schuster2', 'hayden', 'rahman', 'sirinam_vf', 'sirinam_wf']:
for representation in ['rschuster8']:
    for protocol in ['https', 'tor']:
        for platform in ['youtube', 'facebook', 'vimeo', 'rumble']:
            try:
                model = train_model(representation, protocol, platform)
                evaluate_model(model, representation, protocol, platform, results)
            except Exception as e:
                print(representation, protocol, platform, e)
for key, value in results.items():
    print(key, 'test set accuracy is', str(value))
