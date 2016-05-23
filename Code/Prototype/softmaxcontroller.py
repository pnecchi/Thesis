################################################################################
# Description: Softmax controller
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 22 mag 2016 21:54:55 CEST
################################################################################

import numpy as np


class SoftmaxController(object):
    """ Softmax controller for the asset allocation task. """

    def __init__(self, nIn, nOut):
        """ Initialize softmax controller.

        Params:
            nIn (int): input size
            nOut (int): output size
        """
        # Initialize sizes
        self.nIn = nIn
        self.nOut = nOut
        self.nParameters = (nIn + 1) * nOut

        # Initialize controller parameters
        self.parameters = 0.05 * (np.random.rand(self.nParameters) - 0.5)

    def setParameters(self, parameters):
        """ Set the controller parameters.

        Args:
            parameters (np.array): new controller parameters
        """
        self.parameters = parameters

    def activate(self, input):
        """ Activate controller with a certain input.

        Args:
            input (np.array): controller input

        Returns:
            output (np.array): controller output
        """
        # Reshape parameters and add bias
        params = np.reshape(self.parameters, (self.nOut, self.nIn + 1))

        # Add bias to the input
        input_bias = np.append(input, 1.0)

        # Evaluate softmax
        softmax_input = params.dot(input_bias)
        e = np.exp(softmax_input)
        return e / e.sum()
