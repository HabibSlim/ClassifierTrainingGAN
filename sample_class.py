"""
Loads a pretrained model,
generate samples from a given class
"""
import argparse
import functools
import numpy as np
from tqdm import trange
import torch

import utils
import params


def run(config, n_samples, model_name,
        ofile,  y_class,   torch_format):

    # Adjusting batch size for convenience
    G_batch_size = config['G_batch_size']
    if n_samples % G_batch_size != 0:
        print('Defaulting to a batch size of %d.' % 50)
        G_batch_size = 50

    # Initializing generator from configuration
    G = utils.initialize(config, model_name)
    z_ = utils.prepare_z(G_batch_size, G.dim_z,
                         device='cuda', fp16=config['G_fp16'],
                         z_var=config['z_var'])

    n_classes = config['n_classes']

    if y_class is None:
        # Preparing fixed y tensors
        y_ = utils.make_y(G_batch_size, y_class)

        print('Sampling %d images from class %d...'
            % (n_samples, y_class))
    else:
        # Sampling a number of images and save them to an NPZ
        batches_per_class = n_samples/(n_classes*G_batch_size)

        print('Sampling %d images from each class (%d batches per class)...'
            % (n_samples, batches_per_class))

    x, y = [], []
    batch_count = 0
    k = 0
    for i in trange(n_samples / G_batch_size):
        with torch.no_grad():
            if y_class is not None:
                if batch_count == batches_per_class:
                    batch_count = 0
                    k += 1
                else:
                    batch_count += 1
                y_ = utils.make_y(G_batch_size, k)

            images, labels = utils.sample_cond(G, z_, y_)

        # Fetching to CPU
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        # Normalizing for display (optionally)
        if torch_format:
            x += [images]
        else:
            x += [np.uint8(255 * (images + 1) / 2.)]
        y += [labels]

    x = np.concatenate(x, 0)[:n_samples]
    y = np.concatenate(y, 0)[:n_samples]

    print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    npz_filename = '%s/%s.npz' % (config['samples_root'], ofile)
    print('Saving npz to %s...' % npz_filename)
    np.savez(npz_filename, **{'x': x, 'y': y})


def main():
    # Loading configuration
    config = params.params

    # Parsing command line parameters
    parser = argparse.ArgumentParser(description='Parametrized sample generation from GAN weights.')
    parser.add_argument('--num_samples', metavar='nsamples', type=int,
                        nargs=1,
                        default=[10],
                        help='Number of samples to generate '
                             '(default: %(default)s)')
    parser.add_argument('--ofile', metavar='ofile', type=str,
                        nargs=1,
                        default=["samples"],
                        help='Output file name '
                             '(default: %(default)s)')
    parser.add_argument('--model', metavar='model', type=str,
                        nargs=1,
                        help='Model name to use (with weights in ./weights/model_name')
    parser.add_argument('--class', metavar='class', type=int,
                        nargs=1,
                        default=[None],
                        help='Class to sample from (in [O,k-1] for k classes, '
                             'default: sample [num_samples/k] for all classes.)')
    parser.add_argument('--torch_format',
                        action='store_true',
                        help='Save sample archive using the torch format for samples'
                             '(default: False)')     
    args = vars(parser.parse_args())

    # Updating config object
    num_samples = args['num_samples'][0]
    model_name = args['model'][0]
    ofile = args['ofile'][0]
    y_class = args['class'][0]

    # Toggles:
    torch_format = args['torch_format']

    utils.update_config(config)

    run(config, num_samples, model_name, ofile, y_class, torch_format)


if __name__ == '__main__':
    main()
