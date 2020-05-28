"""
Simple wrapper around classifiers used for sample filtering.
"""
import torch

from classifiers import resnet20


class Classifier:
    def __init__(self, model_name, num_classes):
        # Instanciating model
        if model_name == 'resnet20':
            self.model = resnet20(num_classes, 64).to('cuda')
            self.model.eval()
        else:
            print("Unsupported filtering classifier model: %s." % model_name)
            exit(-1)

    def load(self, weights_file):
        """Loading a weights file"""
        self.model.load_state_dict(
            torch.load('./classifiers/weights/%s.pth' % weights_file))

    def filter(self, inputs, labels, threshold):
        """Filtering a batch of images:
          returns a vector True/False values for every input image
        """
        # Getting values in the [0,1] range
        outputs = self.model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        # Getting max and argmax
        probas, preds = torch.max(outputs, dim=1)

        # Correctness of labels & proba above threshold
        mask = (preds == labels) & (probas >= threshold)

        return mask
