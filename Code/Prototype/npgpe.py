################################################################################
# Description: Natural Policy Gradient with Parameter-based Exploration (NPGPE)
#              algorithm, proposed in Miyamae et Al. (2010). The class is based
#              upon the PyBrain architecture.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 22 mag 2016 17:06:58 CEST
################################################################################

import numpy as np


class NPGPE(object):
    """ NPGPE is a gradient estimator technique proposed by Miyamae et Al. in
        "Natural Policy Gradient Methods with Parameter-based Exploration for
        Control Tasks" (2010). This method improves the standard PGPE algorithm
        by using a natural gradient update and is well-suited to infinite
        horizon tasks. It uses optimal baseline and calculate the gradient with
        the log-likelihoods of the hyperparameter distribution.
    """

    # Learning rate
    alphaMu = 0.1
    alphaC = 0.05

    # Standard deviation for parameters initialization
    epsilon = 1.0

    # Discount factor
    gamma = 0.90

    def __init__(self, controller):
        """
        """
        # Initialize module
        self.controller = controller

        # Initialize hyperparameters
        self.nParameters = self.controller.nParameters
        self.mu = np.zeros(self.nParameters)
        self.C = np.diag(self.epsilon * np.ones(self.nParameters))

        # Initialize gradients
        self.gradientMu = np.zeros(self.nParameters)
        self.gradientC = np.zeros((self.nParameters, self.nParameters))

        # Initialize baseline
        self.baseline = None

        # Initialize reward
        self.reward = None


    def _drawControllerParameters(self):
        """ Draw a set of controller parameters from N(mu, C^T C)

        Returns:
            theta (np.array): controller parameters
        """
        # Cache simulation from N(0, I)
        self.x = np.random.randn(self.nParameters)

        # Compute new controllew parameters
        return self.mu + np.dot(self.C.T, self.x)

    def getAction(self, obs):
        """ Select an action given an observation of the system

        Args:
            obs (np.array): observation of the system

        Returns:
            action (np.array): action
        """
        # Draw new controller parameters
        self.theta = self._drawControllerParameters()
        self.controller.setParameters(self.theta)

        # Select action
        action = self.controller.activate(obs)
        return action

    def giveReward(self, r):
        """ Receive a reward

        Args:
            r (double): reward
        """
        self.reward = r

    def computeBaseline(self):
        if self.baseline is None:
            self.baseline = self.reward
        else:
            self.baseline = 0.9 * self.baseline + 0.1 * self.reward

    def calculateGradient(self):
        """ Compute the average reward natural gradient with respect to the
            hyperparameters using the parameter-based policy gradient theorem.

        Returns:
            naturalGradient (np.array): natural gradient estimate
        """
        # Compute gradient wrt means
        newGradientMu = self.theta - self.mu

        # Compute gradient wrt Cholesky factor
        x = np.matrix(self.x)
        Y = x.T.dot(x)
        newGradientC = np.dot(np.triu(Y) - 0.5 * np.diag(np.diag(Y)) -
                              0.5 * np.eye(self.nParameters), self.C)

        # Update gradients
        self.gradientMu = self.gamma * self.gradientMu + newGradientMu
        self.gradientC = self.gamma * self.gradientC + newGradientC

    def learn(self):
        # Compute baseline
        self.computeBaseline()

        # Compute gradients
        self.calculateGradient()

        # Update hyperparameters
        self.mu += self.alphaMu * (self.reward - self.baseline) * self.gradientMu
        self.C += self.alphaC * (self.reward - self.baseline) * self.gradientC


