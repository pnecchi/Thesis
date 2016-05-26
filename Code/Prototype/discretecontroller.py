################################################################################
# Description: Discrete controller for two assets (risk-free and risky)
#              allocation task. The trader can go short (-1), stay neutral (0)
#              or go long (+1) on the risky asset.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        gio 26 mag 2016 10:53:26 CEST
################################################################################

import numpy as np


class DiscreteController(object):

    """ Discrete controller for two assets (risk-free and risky) allocation
        task. The trader can go short (-1), stay neutral (0) or go long (+1)
        on the risky asset. The weight invested on the risk-free asset is given
        by 1 - a_risky, so that the trader invests all of his wealth at each
        time step. """

    def __init__(self, nIn):
        """ Initialize discrete controller.

        Params:
            nIn (int): input size
        """
        # Initialize sizes
        self.nIn = nIn
        self.nParameters = nIn + 1

        # Initialize controller parameters
        self.parameters = 0.01 * (np.random.rand(self.nParameters) - 0.5)

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
        # Add bias to the input
        input_bias = np.append(input, 1.0)

        # Evaluate risky-asset weight
        activation = np.dot(input_bias, self.parameters)
        aRisky = np.sign(activation)
        aFree = 1 - aRisky
        return np.array([aFree, aRisky])
