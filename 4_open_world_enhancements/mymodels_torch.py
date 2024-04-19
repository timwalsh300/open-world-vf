import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from bayesian_torch.layers import Conv1dFlipout, LinearFlipout

# This redefines DFNet from Sirinam et al. "Deep Fingerprinting"
# in PyTorch and in a way that is suitable for tuning a bunch
# of hyperparameters with Ray Tune
#
# It also enables L2 prior regularization for MAP estimation and
# enables Monte Carlo Dropout with the option to sample during
# inference
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

# This defines a temperature scaling layer that has
# just one parameter, the temperature, to scale
# the logits of another trained model
class TemperatureScaling(torch.nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        # initialize T to 1.5
        self.temperature = torch.nn.Parameter(torch.full((1,), 1.5))

    def forward(self, logits):
        return logits / self.temperature

class ConcreteDropout(torch.nn.Module):

    """Concrete Dropout.

    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """Concrete Dropout.

        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = numpy.log(init_min) - numpy.log(1.0 - init_min)
        init_max = numpy.log(init_max) - numpy.log(1.0 - init_max)

        self.p_logit = torch.nn.parameter.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:

        """Calculates the forward pass.

        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.

        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.

        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regulariser * x[0].numel()

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: torch.Tensor) -> torch.Tensor:

        """Computes the Concrete Dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.

        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x

# This redefines the DFNetTunable model using Spike and Slab Dropout
# with Flipout and Concrete Dropout layers in every block
class DFNetTunableSSCD(nn.Module):
    # w and d are only used for training, so they should provide when instantiating
    # the model before training, but can / should be omitted when instantiating the
    # model to load a state dictionary for testing after training 
    def __init__(self, input_shape, classes, hyperparameters, w = 1, d = 1):
        super(DFNetTunableSSCD, self).__init__()
        self.hyperparameters = hyperparameters      
        self.w = w
        self.d = d
        self.conv_padding_left = int(hyperparameters['kernel'] / 2)
        self.pool_padding_left = int(hyperparameters['pool'] / 2)
        self.conv_padding_right = int(hyperparameters['kernel'] / 2) - 1
        self.pool_padding_right = int(hyperparameters['pool'] / 2) - 1
        conv_downsampling = hyperparameters['conv_stride'] ** 8 if hyperparameters['conv_stride'] > 1 else 1
        pool_downsampling = hyperparameters['pool_stride'] ** 4 if hyperparameters['pool_stride'] > 1 else 1

        # Block 1
        self.block1_conv1 = Conv1dFlipout(in_channels=input_shape[0], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act1 = nn.ELU()
        self.block1_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act2 = nn.ELU()
        self.block1_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block1_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Block 2
        self.block2_conv1 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act1 = nn.ELU()
        self.block2_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act2 = nn.ELU()
        self.block2_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block2_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Block 3
        self.block3_conv1 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act1 = nn.ELU()
        self.block3_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act2 = nn.ELU()
        self.block3_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block3_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Block 4
        self.block4_conv1 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act1 = nn.ELU()
        self.block4_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act2 = nn.ELU()
        self.block4_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block4_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = LinearFlipout(in_features=hyperparameters['filters'] * math.ceil(input_shape[1] / (conv_downsampling * pool_downsampling)), out_features=hyperparameters['fc_neurons'])
        self.fc1_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc1_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()
        self.fc1_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        self.fc2 = LinearFlipout(in_features=hyperparameters['fc_neurons'], out_features=hyperparameters['fc_neurons'])
        self.fc2_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc2_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()
        self.fc2_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        self.fc3 = LinearFlipout(in_features=hyperparameters['fc_neurons'], out_features=classes)

    # training = True at all times; we don't even use this argument; it's
    # only here so that we can re-use the same functions that we previously
    # wrote for Monte Carlo Dropout (which does require this argument)
    def forward(self, x, training = True):
        # Block 1
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv1(x, return_kl = False)
        x = self.block1_bn1(x)
        x = self.block1_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv2(x, return_kl = False)
        x = self.block1_bn2(x)
        x = self.block1_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block1_cd(x, self.block1_pool)

        # Block 2
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv1(x, return_kl = False)
        x = self.block2_bn1(x)
        x = self.block2_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv2(x, return_kl = False)
        x = self.block2_bn2(x)
        x = self.block2_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block2_cd(x, self.block2_pool)

        # Block 3
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv1(x, return_kl = False)
        x = self.block3_bn1(x)
        x = self.block3_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv2(x, return_kl = False)
        x = self.block3_bn2(x)
        x = self.block3_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block3_cd(x, self.block3_pool)

        # Block 4
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv1(x, return_kl = False)
        x = self.block4_bn1(x)
        x = self.block4_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv2(x, return_kl = False)
        x = self.block4_bn2(x)
        x = self.block4_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block4_cd(x, self.block4_pool)

        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x, return_kl = False)
        x = self.fc1_bn(x)
        x = self.fc1_cd(x, self.fc1_act)

        x = self.fc2(x, return_kl = False)
        x = self.fc2_bn(x)
        x = self.fc2_cd(x, self.fc2_act)

        x = self.fc3(x, return_kl = False)
        return x
    
    def bernoulli_kl_loss(self, representation):
        # our prior for p_drop is 0.5 to maximize uncertainty
        #
        # here are the number of neurons in each CD layer for the
        # schuster8_tor and dschuster16_https models
        neurons = {'schuster8_tor': [480,120,30,8,128,128],
                   'dschuster16_https': [960,240,60,15,1024,1024]}
        
        bernoulli_kl_loss = 0.0
        bernoulli_kl_loss += neurons[representation][0] * (self.block1_cd.p * torch.log(self.block1_cd.p / 0.5) + (1 - self.block1_cd.p) * torch.log((1 - self.block1_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][1] * (self.block2_cd.p * torch.log(self.block2_cd.p / 0.5) + (1 - self.block2_cd.p) * torch.log((1 - self.block2_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][2] * (self.block3_cd.p * torch.log(self.block3_cd.p / 0.5) + (1 - self.block3_cd.p) * torch.log((1 - self.block3_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][3] * (self.block4_cd.p * torch.log(self.block4_cd.p / 0.5) + (1 - self.block4_cd.p) * torch.log((1 - self.block4_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][4] * (self.fc1_cd.p * torch.log(self.fc1_cd.p / 0.5) + (1 - self.fc1_cd.p) * torch.log((1 - self.fc1_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][5] * (self.fc2_cd.p * torch.log(self.fc2_cd.p / 0.5) + (1 - self.fc2_cd.p) * torch.log((1 - self.fc2_cd.p) / 0.5))
        return bernoulli_kl_loss

# this is just a sanity check for the baseline model
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
        dummy_output = model(dummy_input, training = False)
        print(dummy_output)
