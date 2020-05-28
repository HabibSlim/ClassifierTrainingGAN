'''
Loads a pretrained model and generate samples from it
'''
import argparse
import functools
import numpy as np
from tqdm import trange
import torch

import utils
import params


def run(config, n_samples, model_name, ofile):
    # Initializing the generator from configuration
    G = utils.initialize(config, model_name)

    # Update batch size setting used for G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device='cuda', fp16=config['G_fp16'],
                               z_var=config['z_var'])

    # Sample function
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_)

    # Sample a number of images and save them to an NPZ
    print('Sampling %d random images...' % n_samples)

    x, y = [], []
    for i in trange(int(np.ceil(n_samples / float(G_batch_size)))):
        with torch.no_grad():
            images, labels = sample()
        x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
        y += [labels.cpu().numpy()]
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
    args = vars(parser.parse_args())

    # Updating config object
    num_samples = args['num_samples'][0]
    model_name = args['model'][0]
    ofile = args['ofile'][0]

    utils.update_config(config)

    run(config, num_samples, model_name, ofile)


if __name__ == '__main__':
    main()
