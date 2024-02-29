import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DFNetTunable(nn.Module):
    def __init__(self, input_shape, classes, hyperparameters):
        super(DFNetTunable, self).__init__()
        self.hyperparameters = hyperparameters        
        self.conv_padding_left = int(hyperparameters['kernel'] / 2)
        self.pool_padding_left = int(hyperparameters['pool'] / 2)
        self.conv_padding_right = int(hyperparameters['kernel'] / 2) - 1
        self.pool_padding_right = int(hyperparameters['pool'] / 2) - 1
        conv_downsampling = hyperparameters['conv_stride'] ** 8 if hyperparameters['conv_stride'] > 1 else 1
        pool_downsampling = hyperparameters['pool_stride'] ** 4 if hyperparameters['pool_stride'] > 1 else 1

        # Block 1
        self.block1_conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act1 = nn.ELU()
        self.block1_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act2 = nn.ELU()
        self.block1_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Block 2
        self.block2_conv1 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act1 = nn.ELU()
        self.block2_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act2 = nn.ELU()
        self.block2_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Block 3
        self.block3_conv1 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act1 = nn.ELU()
        self.block3_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act2 = nn.ELU()
        self.block3_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Block 4
        self.block4_conv1 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act1 = nn.ELU()
        self.block4_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act2 = nn.ELU()
        self.block4_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=hyperparameters['filters'] * math.ceil(input_shape[1] / (conv_downsampling * pool_downsampling)), out_features=hyperparameters['fc_neurons'])
        self.fc1_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc1_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()

        self.fc2 = nn.Linear(in_features=hyperparameters['fc_neurons'], out_features=hyperparameters['fc_neurons'])
        self.fc2_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc2_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()

        self.fc3 = nn.Linear(in_features=hyperparameters['fc_neurons'], out_features=classes)

    # 'training' is True when using the model in the train() mode, and when
    # using it in the eval() mode for Monte Carlo Dropout. When using it
    # in eval() for the baseline approach, it is False
    def forward(self, x, training):
        # Block 1
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = torch.nn.functional.dropout(self.block1_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 2
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = torch.nn.functional.dropout(self.block2_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 3
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = torch.nn.functional.dropout(self.block3_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 4
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = torch.nn.functional.dropout(self.block4_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = torch.nn.functional.dropout(self.fc1_act(x),
                                        p = self.hyperparameters['fc_dropout'],
                                        training = training)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = torch.nn.functional.dropout(self.fc2_act(x),
                                        p = self.hyperparameters['fc_dropout'],
                                        training = training)

        x = self.fc3(x)
        return x
        
    def L2reg(self, l2_coeff):
        l2reg_sum = l2_coeff * sum(torch.square(theta).sum() for theta in self.parameters())
        return l2reg_sum

class TemperatureScaling(torch.nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

if __name__ == '__main__':
    CLASSES = 61
    INPUT_SHAPES = {'schuster8_tor': (1, 1920), 'dschuster16_https': (2, 3840)}
    BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
    def print_output_size(module, input, output):
        print(f"{module.__class__.__name__}: {output.shape}")

    for name in ['schuster8_tor', 'dschuster16_https']:
        model = DFNetTunable(INPUT_SHAPES[name], CLASSES, BEST_HYPERPARAMETERS[name])
        print(model)
        for layer in model.modules():
            layer.register_forward_hook(print_output_size)
        model.eval()
        dummy_input = torch.rand(1, *INPUT_SHAPES[name])
        dummy_output = model(dummy_input)
        print(dummy_output)
