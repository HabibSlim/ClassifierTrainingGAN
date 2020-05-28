'''
This file contains various utility functions.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''
from __future__ import print_function
import numpy as np
import time

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as tforms
import BigGAN


# Convenience dicts
dset_dict = {'I32':  dset.ImageNet,
             'I64':  dset.ImageNet,
             'I128': dset.ImageNet,
             'I256': dset.ImageNet,
             'C10':  dset.CIFAR10,
             'C100': dset.CIFAR100}

imsize_dict = {'I32':  32,  'I32_hdf5':  32,
               'I64':  64,  'I64_hdf5':  64,
               'I128': 128, 'I128_hdf5': 128,
               'I256': 256, 'I256_hdf5': 256,
               'C10':  32,  'C100':      32}

root_dict = {'I32':  'ImageNet', 'I32_hdf5':  'ILSVRC32.hdf5',
             'I64':  'ImageNet', 'I64_hdf5':  'ILSVRC64.hdf5',
             'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
             'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
             'C10':  'cifar',    'C100':      'cifar'}

nclass_dict = {'I32':  1000, 'I32_hdf5':  1000,
               'I64':  1000, 'I64_hdf5':  1000,
               'I128': 1000, 'I128_hdf5': 1000,
               'I256': 1000, 'I256_hdf5': 1000,
               'C10':  10,   'C100':      100}

classes_per_sheet_dict = {'I32':  50, 'I32_hdf5':  50,
                          'I64':  50, 'I64_hdf5':  50,
                          'I128': 20, 'I128_hdf5': 20,
                          'I256': 20, 'I256_hdf5': 20,
                          'C10':  10, 'C100':      100}

activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu':         nn.ReLU(inplace=False),
                   'ir':           nn.ReLU(inplace=True)}


# Load a model's weights, optimizer, and the state_dict
def load_weights(model_name, name_suffix=None, G_ema=None, strict=True):
    root = './weights/%s' % model_name
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading GAN weights from %s...' % root)

    if G_ema is not None:
        G_ema.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))),
            strict=strict)


# Loading the test set
def make_test_loader(dset_name, batch_size, transform):
    # TODO: Resize ImageNet if used at some point
    transform = tforms.Compose([tforms.ToTensor(), transform])

    testset = dset_dict[dset_name](root='./data', train=False,
                                   download=True, transform=transform)

    return torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                       shuffle=False, num_workers=2,
                                       pin_memory=True)


# Measuring time in a synchronized manner
def ctime():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return time.time()


# Initializing and loading a BigGAN net
def initialize(config, model_name):
    # Seed RNG
    seed_rng(config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    G = BigGAN.Generator(**config).cuda()
    count_parameters(G)

    # Loading pre-trained BigGAN
    load_weights(model_name,
                 config['load_weights'],
                 G, strict=False)

    # Switching to eval mode
    G.eval()

    return G


# Updating the configuration file
def update_config(config):
    config['resolution'] = imsize_dict[config['dataset']]
    config['n_classes'] = nclass_dict[config['dataset']]
    config['G_activation'] = activation_dict[config['G_nl']]
    config['D_activation'] = activation_dict[config['D_nl']]
    config['skip_init'] = True
    config['no_optim'] = True


# Utility file to seed rngs
def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# Function to join strings or ignore them
# Base string is the string to link "strings," while strings
# is a list of strings or Nones.
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


# Sample function
def sample(G, z_, y_):
    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        G_z = nn.parallel.data_parallel(G, (z_, G.shared(y_)))

        return G_z, y_


# Sample function for class-conditional samples
def sample_cond(G, z_, y_):
    with torch.no_grad():
        z_.sample_()
        G_z = nn.parallel.data_parallel(G, (z_, G.shared(y_)))

        return G_z, y_


# Convenience function to count the number of parameters in a module
def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))


# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
    return torch.randint(low=0, high=num_classes, size=(batch_size,),
                         device=device, dtype=torch.int64, requires_grad=False)


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


# Prepare random z, y
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False, z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses)
    y_ = y_.to(device, torch.int64)
    return z_, y_


# Prepare random z
def prepare_z(G_batch_size, dim_z, device='cuda',
              fp16=False, z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()
    return z_


# Create a fixed y tensor
def make_y(G_batch_size, y_class, device='cuda'):
    y_ = torch.full((G_batch_size,), fill_value=y_class)
    y_ = y_.to(device, torch.int64)

    return y_


# This version of Adam keeps an fp32 copy of the parameters and
# does all of the parameter updates in fp32, while still doing the
# forwards and backwards passes using fp16 (i.e. fp16 copies of the
# parameters and fp16 activations).
#
# Note that this calls .float().cuda() on the params.
import math
from torch.optim.optimizer import Optimizer


class Adam16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

    # Safety modification to make sure we floatify our state
    def load_state_dict(self, state_dict):
        super(Adam16, self).load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
                self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
                self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                    # Fp32 copy of the weights
                    state['fp32_p'] = p.data.float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], state['fp32_p'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
                p.data = state['fp32_p'].half()

        return loss
