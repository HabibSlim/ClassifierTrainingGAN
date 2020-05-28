"""
Custom dataloader using classifier filtering of generated samples.
"""
import random
import torch
from torch.utils.data import DataLoader


# Utility functions
def new_g_pair():
    im = torch.tensor([]).to('cuda')
    y  = torch.tensor([]).long().to('cuda')

    return im, y


class FilteredLoader(DataLoader):
    """General FilteredLoader"""
    def __init__(self,
                 gen_fn,  cls_fn,
                 n_class, n_batch,
                 threshold,
                 transform,
                 batch_size,
                 num_workers,
                 fixed_ds):
        """gen_fn:    conditional sample generator function
           cls_fn:    classifier function
           n_class:   number of classes to sample from (=10 if CIFAR-10)
           n_batch:   number of batches per class to generate
           threshold: probability threshold for filtering
           transform: transformation to apply to batches
           fixed_ds:  set to true if the dataset must be the same across every
                      epoch (generates the dataset the first time, loads it from
                             memory afterwards)
        """
        super().__init__(dataset=None,
                         batch_size=batch_size, 
                         num_workers=num_workers)

        # Initializing threshold tensor
        self.thr = torch.tensor([threshold for _ in range(batch_size)]) \
                        .to('cuda')                                

        # Generator and filtering functions
        self.gen_fn = gen_fn
        self.cls_fn = cls_fn

        self.n_class = n_class
        self.n_batch = n_batch

        # Counting number of filtered samples
        self.filter_count = 0
        self.batch_count = 0

        # Creating tensor transformation
        mean = torch.as_tensor(transform[0])
        std  = torch.as_tensor(transform[1])

        self.mean = mean[:, None, None]
        self.std  = std[:, None, None]

        # Keeping track of the previously generated dataset
        self.fixed_ds = fixed_ds
        self._cached = False

        if fixed_ds:
            print("[FilterLoader] Using fixed dataset loader...")
            # Storing generated images
            self._stored_im, self._stored_y = new_g_pair()
            # Initializing random batches
            self._indexes = list(range(self.train_length()))

    def _update_count(self):
        """Updating batches per class count"""
        self.batch_count += 1
        if self.batch_count > self.n_batches():
            self._cached = True
            raise StopIteration

    def _shuffle_samples(self):
        """Randomly shuffling stored samples"""
        random.shuffle(self._indexes)

    def _gen_indexes(self):
        """Fetching #batch_size random indices"""
        batch_idx = self._indexes[:self.batch_size]

        # Sampling without replacement for faster convergence
        self._indexes = self._indexes[self.batch_size:]

        return batch_idx

    def __next__(self):
        self._update_count()

        # Generating/filtering until enough for a batch
        ims, ys = new_g_pair()

        if self.fixed_ds and self._cached:
            batch_idx = self._gen_indexes()

            ims = self._stored_im[batch_idx]
            ys  = self._stored_y[batch_idx]
        else:
            with torch.no_grad():
                while ims.shape[0] < self.batch_size:
                    # Generating batch and normalizing
                    batch = self.gen_fn()

                    # Applying transformation
                    # self._norm(batch)
                    inputs, labels = batch[0], batch[1]

                    # Applying filter mask
                    mask = self.cls_fn(inputs, labels, self.thr)
                    inputs = inputs[mask]
                    labels = labels[mask]

                    # Adding images/labels to the batch
                    ims = torch.cat((ims, inputs))
                    ys  = torch.cat((ys,  labels))

                    self.filter_count += self.batch_size - inputs.shape[0]

                    # Trimming excess samples
                    if ims.shape[0] > self.batch_size:
                        ims = torch.split(ims, self.batch_size)[0]
                        ys  = torch.split(ys,  self.batch_size)[0]

            if self.fixed_ds:
                self._stored_im = torch.cat((self._stored_im, ims))
                self._stored_y  = torch.cat((self._stored_y,  ys))

        return ims, ys

    def __iter__(self):
        return self

    def _norm(self, batch):
        """Mean/STD normalization on batch"""
        batch.sub_(self.mean).div_(self.std)

    def reset(self):
        """Reset loader between each epoch"""
        self.batch_count = 0
        self.filter_count = 0
        if self.fixed_ds:
            self._shuffle_samples()

    def filter_ratio(self):
        ratio = self.filter_count / self.train_length()
        return ratio * 100

    def n_batches(self):
        return self.n_class * self.n_batch

    def train_length(self):
        return self.batch_size * self.n_batches()
