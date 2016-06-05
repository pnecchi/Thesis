################################################################################
# Description: Module containing various actors for the asset allocation problem
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 05 giu 2016 17:51:25 CEST
################################################################################

import numpy as np


def Actor(object):
    """ Actor class which specifies the generic interface of an actor. """

    def __init__(self, dimIn, dimOut):
        """ Initialize actor.

        Args:
            dimIn (int): state size
            dimOut (int): action size
        """

        # Initialize input size, i.e. size of the state
        self.dimIn = dimIn

        # Initialize output size, i.e. size of the action
        self.dimOut = dimOut

    def selectAction(self, state):
        """ Select an action given a state. This function correspond to the
        policy employed by the actor and it can be either stochastic or
        deterministic.

        Args:
            state (np.array): state

        Returns:
            action (np.array): selected action, i.e. a ~ pi(s)
        """
        pass

    def scoreFunction(self, state, action):
        """ Compute the policy score function associated to a given state and
        action, i.e. nabla_theta log pi_theta(s,a)

        Args:
            state (np.array): state
            action (np.array): action

        Returns:
            score (np.array): score function
        """
        pass


class BoltzmannExploration(Actor):
    """ Boltzmann exploring agent for discrete action space, cf. Van Otterloo
    and Wiering - Reinforcement Learning State-Of-The-Art (2012), chapter 7. The
    features to be used are passed as input."""

    def __init__(self, dimIn, dimOut, actionList, features):
        """ Initialize the BoltzmannExploration Actor.

        Args:
            dimIn (int): state size
            dimOut (int): action size
            actionList (np.array): possible action values (size dimOut)
            features (np.array): features vector Phi_pi(s,a) (size dimPar)
        """
        # Initialize Actor base class
        Actor.__init__(self, dimIn, dimOut)

        # Initialize action list
        self._actionList = actionList

        # Initialize actor features
        self._features = features
        self._dimPar = features.size()

        # Initialize actor parameters
        self._parameters = 0.05 * np.random.randn()

    def _evaluateBoltzmannProbabilities(self, state):
        """ Evaluate the Boltzmann probabilities associated to each action for
        the given state. The probabilities are stored in the private variable
        self._actionProb.

        Args:
            state (np.array): state
        """
        # Cache state
        self._lastState = state

        # Evaluate feature vectors for all possible actions and cache result
        self._featuresEval = np.array([self._features(state, action)
                                    for action in self._actionList])

        # Evaluate probabilities of selecting each action
        expFeatures = np.exp(np.dot(self._featuresEval, self._parameters.T))
        self._actionProb = expFeatures / np.sum(expFeatures)

    def selectAction(self, state):
        """ Given a state, sample an action from the Boltzmann probability
        distribution with the current parameters.

        Args:
            state (np.array): state

        Returns:
            action (np.array): selected action, i.e. a ~ pi(s)
        """
        # Check if the results for the state are cached, otherwise evaluate
        # Boltzmann probabilities for each action
        if state != self._lastState:
            self._evaluateBoltzmannProbabilities(state)

        # Sample action from Boltzmann probability distribution
        self._action = np.random.choice(self._actionList, p=self._actionProb)
        return self._action

    def scoreFunction(self, state, action):
        """ Given a state and an action, compute the likelihood score vector.

        Args:
            state (np.array): state
            action (np.array): action

        Return:
            score (np.array): likelihood score for the given state and action
        """
        # Check if the results for the state are cached, otherwise evaluate
        # Boltzmann probabilities for each action
        if state != self._lastState:
            self._evaluateBoltzmannProbabilities(state)

        # Compute likelihood score
        score = self._featuresEval[self._actionList == action, :] - \
            np.dot(self._featuresEval, self._actionProb.T)
        return score
