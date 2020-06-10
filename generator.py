"""
Simple wrapper around generators.
"""
import utils
import random


class GeneratorWrapper:
    def __init__(self, config, model_name, thr=None, multi_gans=None, gan_weights=None):
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
        self.gan_weights = gan_weights

        # Preparing sampling functions
        self.z_, self.y_ = utils.prepare_z_y(G_batch_size, config['dim_z'],
                                            n_classes,
                                            device='cuda',
                                            fp16=config['G_fp16'],
                                            z_var=config['z_var'],
                                            thr=thr)

        # Preparing fixed y tensors
        self.y_fixed = {y: utils.make_y(G_batch_size, y) for y in range(n_classes)}

    def get_g(self):
        """Selecting a generator at random"""
        if self.multi_gans is None:
            return self.G
        elif self.gan_weights is not None:
            return random.choices(self.G, self.gan_weights)[0]
        else:
            return random.choice(self.G)

    def gen_batch(self, _=None):
        """Generate a batch of random samples using G"""
        return utils.sample(self.get_g(), self.z_, self.y_)

    def gen_batch_cond(self, class_y):
        """Generate a batch of samples of class y using G"""
        class_tensor = self.y_fixed[class_y]
        return utils.sample_cond(self.get_g(), self.z_, class_tensor)
