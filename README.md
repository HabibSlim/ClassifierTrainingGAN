Generating from BigGAN, classifier filtering and training.

# Usage
## Training a classifier

Training a classifier requires:
- pre-trained classifier weights in `classifier/weights/model_name.pth`
- pre-trained BigGAN weights in `weights/weights_folder_name/`

To run the script:

`python3 train_classifier.py [options]`

Positional arguments are as follows:

- `num_batches`: number of batches per class to train the classifier with
- `batch_size`: size of each batch (same for generation/filtering/training)
- `model`: weights file to use for the GAN
- `classifier_model`: weights file to use for the filtering classifier
- `ofile`: output file name
- `threshold`: threshold probability for filtering using the clasifier
- `num_workers`: number of workers for the dataloader
- `epochs`: number of epochs for the full training session

## Sampling from GAN weights

Two other scripts have been written to sample randomly (sample.py), and to generate samples conditioned by an input class (sample_class.py).

# Scripts

Some bash scripts are already in the folder `./scripts/` to run some tests.