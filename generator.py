"""
Simple wrapper around generators.
"""
import utils


class GeneratorWrapper:
    def __init__(self, config, model_name):
        # Initializing generator from configuration
        self.G = utils.initialize(config, model_name)

        # Update batch size setting used for G
        G_batch_size = config['G_batch_size']

        # Preparing sampling functions
        n_classes = config['n_classes']
        self.z_, self.y_ = utils.prepare_z_y(G_batch_size, self.G.dim_z,
                                             n_classes,
                                             device='cuda',
                                             fp16=config['G_fp16'],
                                             z_var=config['z_var'])

        # Preparing fixed y tensors
        self.y_fixed = {y: utils.make_y(G_batch_size, y) for y in range(n_classes)}

    def gen_batch(self):
        """Generate a batch of random samples using G"""
        return utils.sample(self.G, self.z_, self.y_)

    def gen_batch_cond(self, class_y):
        """Generate a batch of samples of class y using G"""
        class_tensor = self.y_fixed[class_y]

        return utils.sample_cond(self.G, self.z_, class_tensor)
