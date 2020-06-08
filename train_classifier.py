"""
Loads a pretrained GAN, generates samples from it,
filters generated samples using a pretrained classifier,
and trains a classifier using the generated samples.
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

import utils
import params
from tqdm.auto import tqdm as tqdm

from classifiers import Classifier, resnet20
from generator import GeneratorWrapper
from filter_loader import FilteredLoader


# Default normalization for CIFAR-10 samples
norm_vals = ([0.5, 0.5, 0.5],  # mean
             [0.5, 0.5, 0.5])  # std

# Debugging generated batches
CHECK_BATCH = True

def run(config,     num_batches,      batch_size,
        model_name, class_model_name, ofile,
        threshold,  num_workers,      epochs,
        multi_gans, gan_weights,      trunc_norm,
        fixed_dset, transform,        filter_samples):

    # Instanciating generator
    config['G_batch_size'] = batch_size

    generator = GeneratorWrapper(config, model_name, trunc_norm, multi_gans, gan_weights)
    generator_fn = generator.gen_batch
    if gan_weights:
        print('Using GAN weights (multi-GAN setting).')

    # Instanciating filtering classifier
    if filter_samples:
        print('Using ResNet20 weights: %s.pth' % class_model_name)
        filter_net = Classifier('resnet20', config['n_classes'])
        filter_net.load(class_model_name)
        filter_fn = filter_net.filter
    else:
        filter_fn = None

    # Creating a filtered loader using the classifier
    num_classes = config['n_classes']
    loader = FilteredLoader(generator_fn,
                            filter_fn,
                            num_classes,
                            num_batches,
                            batch_size,
                            threshold,
                            num_workers,
                            fixed_dset,
                            transform,
                            norm_vals)

    print('Training using %d generated images per epoch'
          % loader.train_length())

    # Creating a blank ResNet
    net = resnet20(config['n_classes'], width=64).to('cuda')

    # Initializing loss functions, optimizer, learning rate scheduler
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

    # Evaluating the model on the test set
    test_loader = utils.make_test_loader(config['dataset'],
                                         batch_size,
                                         transforms.Normalize(*norm_vals))

    # Training the model
    t1 = utils.ctime()
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch: %3d' % (epoch+1), end="  ")

        train(net, loader, batch_size, optimizer, cross_entropy)
        scheduler.step()

        acc = evaluate(net, test_loader)
        best_acc = max(acc, best_acc)
        loader.reset()
        print('Val acc: %4.2f %% ' % evaluate(net, test_loader),
              ' | Best acc: %4.2f %%\n' % best_acc)

    tt = utils.ctime() - t1
    print('Finished training, total time: %4.2fs' % tt)
    print('Best accuracy achieved: %4.5f %%' % best_acc)

    # Saving output model
    output = './output/%s.pth' % ofile
    print('Saving trained classifier in %s' % output)
    torch.save(net.state_dict(), output)


def train(net, loader, batch_size, optimizer, loss_fn):
    """Training a classifier for one epoch"""
    global CHECK_BATCH

    train_length = loader.train_length()
    batches_length = loader.n_batches()
    running_loss = 0.0
    net.train()

    with tqdm(total=batches_length) as progress_bar:
        correct = 0
        total = 0

        for data in loader:
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Saving one filtered batch to debug output
            if CHECK_BATCH:
                npz_filename = 'check_batch.npz'
                inputs_scaled = np.uint8(255 * (inputs.cpu().numpy() + 1) / 2.)
                np.savez(npz_filename, **{'x': inputs_scaled, 'y': labels.cpu()})
                CHECK_BATCH = False

            progress_bar.update(1)

    # Computing train loss/accuracy
    print('Train loss: %5.4f' % ((running_loss * batch_size) / train_length),
          'Train acc: %4.2f' % (100 * correct / total))


def evaluate(net, test_loader):
    """Evaluating the resulting classifier using a given test set loader"""
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    # Loading configuration
    config = params.params

    # Parsing command line parameters
    parser = argparse.ArgumentParser(description='Parametrized sample generation from GAN weights.')

    # -> Training parameters:
    parser.add_argument('--num_batches', metavar='nbatches', type=int,
                        nargs=1,
                        default=[1],
                        help='Number of batches per class to train the classifier with '
                             '(default: %(default)s)')
    parser.add_argument('--batch_size', metavar='batch_size', type=int,
                        nargs=1,
                        default=[64],
                        help='Size of each batch (same for generation/filtering/training, '
                             'default: %(default)s)')
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        nargs=1,
                        default=[10],
                        help='Number of epochs to train the classifier for '
                             '(default: %(default)s)')

    # -> Input/Output:
    parser.add_argument('--model', metavar='model', type=str,
                        nargs=1,
                        help='Weights file to use for the GAN (of the form: ./weights/model_name.pth)')
    parser.add_argument('--classifier_model', metavar='class_model', type=str,
                        nargs=1,
                        default=['resnet20'],
                        help='Weights file to use for the filtering classifier (of the form: ./classifiers/weights/class_model_name.pth)')
    parser.add_argument('--ofile', metavar='ofile', type=str,
                        nargs=1,
                        default=["trained_net"],
                        help='Output file name '
                             '(default: %(default)s)')
    parser.add_argument('--num_workers', metavar='num_workers', type=float,
                        nargs=1,
                        default=[1],
                        help='Number of workers to use for the dataloader.'
                             '(default: %(default)s)')

    # -> Methods and parameters:
    parser.add_argument('--threshold', metavar='threshold', type=float,
                        nargs=1,
                        default=[0.9],
                        help='Threshold probability for filtering '
                             '(default: %(default)s)')
    parser.add_argument('--truncate', metavar='truncate', type=float,
                        nargs=1,
                        default=[None],
                        help='Sample latent z from a truncated normal '
                             '(default: no truncation).')
    parser.add_argument('--fixed_dset',
                        action='store_true',
                        help='Use a fixed generated dataset for training '
                             '(of size: batch_size x num_batches x num_classes, '
                             'default: False)')
    parser.add_argument('--transform',
                        action='store_true',
                        help='Apply image transformations to generated images '
                             '(default: False)')
    parser.add_argument('--filter_samples',
                        action='store_true',
                        help='Enable classifier-filtering of generated images '
                             '(default: False)')

    # -> Multi-GANs stuff:
    parser.add_argument('--multi_gans', metavar='multi_gans', type=int,
                        nargs=1,
                        default=[None],
                        help='Sample using multiple GANs '
                             '(default: %(default)s)')
    parser.add_argument('--gan_weights', metavar='gan_weights', type=float,
                        nargs='+',
                        default=[None],
                        help='Specify weights for each GAN '
                             '(default: sample from each GAN with equiprobability)')
    args = vars(parser.parse_args())

    # Values:
    num_batches = args['num_batches'][0]
    batch_size  = args['batch_size'][0]
    model_name  = args['model'][0]
    class_model_name = args['classifier_model'][0]
    ofile       = args['ofile'][0]
    threshold   = args['threshold'][0]
    num_workers = args['num_workers'][0]
    epochs      = args['epochs'][0]
    trunc_norm  = args['truncate'][0]
    multi_gans  = args['multi_gans'][0]
    gan_weights = args['gan_weights']

    if gan_weights[0] is not None and multi_gans != len(gan_weights):
        print('The list of GAN weights should specify weights for each GAN.')

    # Toggles:
    fixed_dset     = args['fixed_dset']
    transform      = args['transform']
    filter_samples = args['filter_samples']

    # Updating config object
    utils.update_config(config)

    run(config,
        num_batches,
        batch_size,
        model_name,
        class_model_name,
        ofile,
        threshold,
        num_workers,
        epochs,
        multi_gans,
        gan_weights,
        trunc_norm,
        fixed_dset,
        transform,
        filter_samples)


if __name__ == '__main__':
    main()
