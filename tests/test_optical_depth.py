import numpy as np 
import tools21cm as t2c 

def test_Thomson_optical_depth():
    """	
    To calculate the optical depth for a scenario where the Universe is instantaneously
    reionized:

    >>> z_reion = 11.
    >>> redshifts = np.linspace(z_reion, 1100., 50)
    >>> ionfracs = np.zeros(len(redshifts))
    >>> tau0, tau_z = tau(ionfracs, redshifts)
    >>> print 'Total tau: ', tau0[-1]
    0.0884755058758
    """
    z_reion = 11.
    redshifts = np.linspace(z_reion, 1100., 50)
    ionfracs = np.zeros(len(redshifts))
    tau0, tau_z = t2c.tau(ionfracs, redshifts)
    assert np.abs(tau0[-1]-0.0884755058758)<0.01