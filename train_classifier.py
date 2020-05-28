"""
Loads a pretrained GAN, generates samples from it,
filters generated samples using a pretrained classifier,
and trains a classifier using the generated samples.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
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
T = ([0.5, 0.5, 0.5],  # mean
     [0.5, 0.5, 0.5])  # std

# T = ([0.4914, 0.4822, 0.4465],  # mean
#      [0.2470, 0.2435, 0.2616])  # std

# Debugging generated batches
CHECK_BATCH = True


def run(config, num_batches, batch_size, model_name,
        class_model_name, ofile, thr, num_workers, epochs, fixed_dset):

    # Instanciating generator
    config['G_batch_size'] = batch_size
    generator = GeneratorWrapper(config, model_name)

    # Instanciating filtering classifier
    print('Using ResNet20 weights: %s.pth' % class_model_name)
    filter_net = Classifier('resnet20', config['n_classes'])
    filter_net.load(class_model_name)

    # Creating a filtered loader using the classifier
    num_classes = config['n_classes']
    loader = FilteredLoader(generator.gen_batch,
                            filter_net.filter,
                            num_classes,
                            num_batches,
                            thr,
                            T,
                            batch_size,
                            num_workers,
                            fixed_dset)

    print('Training using %d generated images per epoch'
          % loader.train_length())

    # Creating a blank ResNet
    net = resnet20(config['n_classes'], width=64).to('cuda')

    # Initializing loss functions, optimizer, learning rate scheduler
    cross_entropy = nn.CrossEntropyLoss()
    bce_logits    = nn.BCEWithLogitsLoss(pos_weight=18*torch.ones([num_classes]).to('cuda'))
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, (3 * epochs) // 4])

    # Evaluating the model on the test set
    test_loader = utils.make_test_loader(config['dataset'], batch_size, transforms.Normalize(*T))

    # Training the model
    t1 = utils.ctime()
    for epoch in range(epochs):
        print('Epoch: %3d' % (epoch+1), end="  ")

        train(net, loader, batch_size, optimizer, cross_entropy, bce_logits)
        scheduler.step()
        
        print('Filtered samples: %d (%.1f%% of virtual dset size)' % (loader.filter_count, loader.filter_ratio()))
        
        # Validating (on test set for now)
        print('Validation accuracy: %d %% \n' % evaluate(net, test_loader))
        loader.reset()

    tt = utils.ctime() - t1
    print('Finished training, total time: %4.2fs' % tt)
    print('Test accuracy: %d %%' % evaluate(net, test_loader))

    # Saving output model
    output = './output/%s.pth' % ofile
    print('Saving trained classifier in %s' % output)
    torch.save(net.state_dict(), output)


def train(net, loader, batch_size, optimizer, loss_fn_a, loss_fn_b, alpha=0.9, num_classes=10):
    """Training a classifier for one epoch"""
    global CHECK_BATCH

    train_length = loader.train_length()
    batches_length = loader.n_batches()
    running_loss = 0.0
    net.train()

    with tqdm(total=batches_length) as progress_bar:
        for data in loader:
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)

            if alpha <= 0:
                loss = loss_fn_a(outputs, labels)
            elif alpha >= 1:
                loss = loss_fn_b(outputs, F.one_hot(labels, num_classes).type_as(outputs))
            else:
                loss = (1-alpha)*loss_fn_a(outputs, labels) + \
                alpha*loss_fn_b(outputs, F.one_hot(labels, num_classes).type_as(outputs))

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

    # Computing average loss
    print('Train loss: %5.4f' % ((running_loss * batch_size) / train_length))


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

    return (100 * correct / total)


def main():
    # Loading configuration
    config = params.params

    # Parsing command line parameters
    parser = argparse.ArgumentParser(description='Parametrized sample generation from GAN weights.')
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
    parser.add_argument('--threshold', metavar='threshold', type=float,
                        nargs=1,
                        default=[0.9],
                        help='Threshold probability for filtering '
                             '(default: %(default)s)')
    parser.add_argument('--num_workers', metavar='num_workers', type=float,
                        nargs=1,
                        default=[1],
                        help='Number of workers to use for the dataloader.'
                             '(default: %(default)s)')
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        nargs=1,
                        default=[10],
                        help='Number of epochs to train the classifier for '
                             '(default: %(default)s)')
    parser.add_argument('--fixed_dset', metavar='epochs', type=int,
                        nargs=1,
                        default=[False],
                        help='Use a fixed generated dataset for training '
                             '(of size: batch_size x num_batches x num_classes, '
                             'default: %(default)s)')
    args = vars(parser.parse_args())

    num_batches = args['num_batches'][0]
    batch_size  = args['batch_size'][0]
    model_name  = args['model'][0]
    class_model_name = args['classifier_model'][0]
    ofile       = args['ofile'][0]
    threshold   = args['threshold'][0]
    num_workers = args['num_workers'][0]
    epochs      = args['epochs'][0]
    fixed_dset  = args['fixed_dset'][0]

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
        fixed_dset)


if __name__ == '__main__':
    main()
