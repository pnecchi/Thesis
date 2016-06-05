################################################################################
# Description: Module containing various critic implementations
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 05 giu 2016 18:24:01 CEST
################################################################################

import numpy as np


class Critic(object):
    """ Critic class which specifies the generic interface of a critic. """

    def __init__(self, dimIn):
        """ Initialize critic.

        Args:
            dimIn (int): state size
        """
        # Initialize input size, i.e. size of the state
        self.dimIn = dimIn

    def __call__(self, state):
        """ Evaluate a given state.

        Args:
            state (np.array): state to be evaluated

        Returns:
            value (float): state value
        """
        pass


class LinearCritic(Critic):
    """ Critic that uses a linear function approximation """

    def __init__(self, dimIn, features):
        """ Initialize LinearCritic.

        Args:
            dimIn (int): state size
            features (object): features Phi(s)

        """
        # Initialize Critic base class
        Critic.__init__(self, dimIn)

        # Initialize features
        self._features = features
        self._dimPar = features.size()

        # Initialize critic parameters
        self._parameters = 0.05 * np.random.randn()

    def __call__(self, state):
        """ Evaluate a given state.

        Args:
            state (np.array): state to be evaluated

        Returns:
            value (float): state value
        """
        # Cache state
        self._lastState = state

        # Evaluate features and cache result
        self._featuresEval = self._features(state)

        # Evaluate state
        return np.dot(self._featuresEval, self._parameters.T)

    def gradient(self, state):
        """ Compute critic gradient.

        Args:
            state (np.array): state

        Returns:
            gradient (np.array): critic gradient
        """
        if state != self._lastState:
            self._featuresEval = self.features(state)
        return self._featuresEval

    def getParameters(self):
        """ Return critic parameters.

        Returns:
            parameters (np.array): actor parameters
        """
        return self._parameters

    def setParameters(self, parameters):
        """ Set critic parameters.

        Args:
            parameters (np.array): new actor parameters
        """
        self._parameters = parameters
