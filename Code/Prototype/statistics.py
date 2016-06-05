################################################################################
# Description: Some utility classes to keep trace of some useful statistics
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 05 giu 2016 12:08:41 CEST
################################################################################

class ExponentialMovingAverage(object):
    """ Exponential moving average """

    def __init__(self, scheduleConst=1.0, scheduleExp=0.0):
        """ Initialize the exponential moving average. The learning rate is
            scheduleConst / N**scheduleExp.

        Args:
            scheduleConst (float): schedule constant (default 1.0)
            scheduleExp (float): schedule exponent (default 0.0)
        """
        # Initialize learning rate
        self._learningRate = scheduleConst
        self._scheduleExp = scheduleExp

        # Initialize number of observation
        self._N = 0

        # Initialize exponential moving average
        self._ema = None

    def update(self, x):
        """ Add value to the exponential moving average.

        Args:
            x (float): new value
        """
        if self._ema is None:
            self._ema = x
        else:
            self._ema += self._learningRate / (self._N ** self._scheduleExp) * \
                         (x - self._ema)
        self._N += 1

    def get(self):
        return self._ema
