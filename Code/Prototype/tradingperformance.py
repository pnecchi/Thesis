################################################################################
# Description: Various functions to compute trading performance measures, such
#              as portoflio returns, cumulative returns, Sharpe ratio, etc.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 15 mag 2016 13:59:06 CEST
################################################################################

import numpy as np


def portfolioSimpleReturn(assetReturns,
                          oldAllocation,
                          newAllocation,
                          deltaP=0.0,
                          deltaF=0.0,
                          deltaS=0.0):
    """ Compute portfolio simple return including transaction costs over a
    single time interval (t, t+1)

    Args:
        assetReturns (np.array): assets simple returns over (t, t+1)
        oldAllocation (np.array): porfolio allocation at time t
        newAllocation (np.array): new portfolio allocation held in (t, t+1)
        deltaP (double): proportional transaction costs rate
        deltaF (double): fixed transaction cost rate
        deltaS (double): short selling borrowing cost rate

    Returns:
        ptfSimpleReturn (double): portfolio simple return over (t, t+1)
    """


    posDifference = newAllocation - oldAllocation
    tcProp = deltaP * np.sum(np.absolute(posDifference))
    tcShort = deltaS * np.sum(newAllocation.clip(max=0.))
    ptfSimpleReturn = np.dot(newAllocation, assetReturns) - \
        deltaP * np.sum(np.absolute(posDifference)) + \
        deltaS * np.sum(newAllocation.clip(max=0.)) - \
        deltaF * np.any(posDifference == 0)
    return ptfSimpleReturn


def portfolioLogReturn(assetReturns,
                       oldAllocation,
                       newAllocation,
                       deltaP=0.0,
                       deltaF=0.0,
                       deltaS=0.0):
    """ Compute portfolio log-return including transaction costs over a single
    time interval (t, t+1)

    Args:
        assetReturns (np.array): assets simple returns over (t, t+1)
        oldAllocation (np.array): porfolio allocation at time t
        newAllocation (np.array): new portfolio allocation held in (t, t+1)
        deltaP (double): proportional transaction costs rate
        deltaF (double): fixed transaction cost rate
        deltaS (double): short selling borrowing cost rate

    Returns:
        ptfLogReturn (double): portfolio log-return over (t, t+1)
    """
    ptfSimpleReturn = portfolioSimpleReturn(assetReturns,
                                            oldAllocation,
                                            newAllocation,
                                            deltaP, deltaF, deltaS)
    ptfLogReturn = np.log(1.0 + ptfSimpleReturn)
    return ptfLogReturn
