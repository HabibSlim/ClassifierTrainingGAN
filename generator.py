"""
Simple wrapper around generators.
"""
import utils
import random


class GeneratorWrapper:
    def __init__(self, config, model_name, thr=None, multi_gans=None):

        # Updating settings
        G_batch_size = config['G_batch_size']
        n_classes    = config['n_classes']

        # Loading GAN weights
        if multi_gans is None:
            self.G = utils.initialize(config, model_name)
        else:
            # Assuming that weight files follows the naming convention:
            # model_name_k, where k is in [0,multi_gans-1]
            self.G = [utils.initialize(config, model_name + "_%d" % k)
                      for k in range(multi_gans)]
        self.multi_gans = multi_gans

        # Preparing sampling functions
        self.z_, self.y_ = utils.prepare_z_y(G_batch_size, config['dim_z'],
                                            n_classes,
                                            device='cuda',
                                            fp16=config['G_fp16'],
                                            z_var=config['z_var'],
                                            thr=thr)

        # Preparing fixed y tensors
        self.y_fixed = {y: utils.make_y(G_batch_size, y) for y in range(n_classes)}

    def gen_batch(self):
        """Generate a batch of random samples using G"""
        if self.multi_gans is None:
            return utils.sample(self.G, self.z_, self.y_)
        else:
            # Selecting a generator at random
            g = random.choice(self.G)
            return utils.sample(g, self.z_, self.y_)

    def gen_batch_cond(self, class_y):
        """Generate a batch of samples of class y using G"""
        class_tensor = self.y_fixed[class_y]

        if self.multi_gans is None:
            return utils.sample_cond(self.G, self.z_, class_tensor)
        else:
            # Selecting a generator at random
            g = random.choice(self.G)
            return utils.sample_cond(g, self.z_, class_tensor)
