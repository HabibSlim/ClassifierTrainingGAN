"""
Loads a pretrained model,
generate samples from a given class
"""
import argparse
import numpy as np
from tqdm import trange

import utils
import params
from generator import GeneratorWrapper


def run(config,    n_samples,  model_name,
        ofile,     y_class,    torch_format,
        transform, multi_gans, trunc_norm):

    # Adjusting batch size for convenience
    if n_samples % config['batch_size'] != 0:
        print('Defaulting to a batch size of %d.' % 50)
        config['batch_size'] = 50
        config['G_batch_size'] = config['batch_size']

    # Initializing generator from configuration
    generator = GeneratorWrapper(config, model_name, trunc_norm, multi_gans)
    sample_fn = generator.gen_batch_cond

    n_classes = config['n_classes']

    k = 0
    if y_class is not None:
        k = y_class

        print('Sampling %d images from class %d...'
            % (n_samples, y_class))
    else:
        # Sampling a number of images and save them to an NPZ
        batches_per_class = n_samples/(n_classes*config['batch_size'])

        print('Sampling %d images from each class (%d batches per class)...'
            % (n_samples/n_classes, batches_per_class))

    if transform:
        print('Using data transformations')
        T = utils.load_transform()

    x, y = [], []
    for b in trange(1, int(n_samples / config['batch_size'])+1):
        images, labels = sample_fn(k)

        # Fetching to CPU
        images = images.cpu()
        labels = labels.cpu()

        # Applying transformations
        if transform:
            for i,im in enumerate(images):
                im = (im * 0.5 + 0.5).clamp_(0, 1)
                images[i] = T(im)

        images = images.numpy()
        labels = labels.numpy()

        # Normalizing for display (optionally)
        if torch_format:
            x += [images]
        else:
            x += [np.uint8(255 * (images + 1) / 2.)]
        y += [labels]

        # Updating current class
        if y_class is None and b % batches_per_class == 0:
            k += 1

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
    parser.add_argument('--multi_gans', metavar='multi_gans', type=int,
                        nargs=1,
                        default=[None],
                        help='Sample using multiple GANs '
                             '(default: %(default)s)')
    parser.add_argument('--truncate', metavar='truncate', type=float,
                        nargs=1,
                        default=[None],
                        help='Sample latent z from a truncated normal '
                             '(default: no truncation).')

    parser.add_argument('--torch_format',
                        action='store_true',
                        help='Save sample archive using the torch format for samples'
                             '(default: False)')
    parser.add_argument('--transform',
                        action='store_true',
                        help='Apply image transformations to generated images '
                             '(default: False)')
    args = vars(parser.parse_args())

    # Values:
    num_samples = args['num_samples'][0]
    model_name  = args['model'][0]
    ofile       = args['ofile'][0]
    y_class     = args['class'][0]
    multi_gans  = args['multi_gans'][0]
    trunc_norm  = args['truncate'][0]

    # Toggles:
    torch_format = args['torch_format']
    transform    = args['transform']

    # Updating config object
    utils.update_config(config)

    run(config,
        num_samples,
        model_name,
        ofile,
        y_class,
        torch_format,
        transform,
        multi_gans,
        trunc_norm)


if __name__ == '__main__':
    main()
