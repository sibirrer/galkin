import math
import numpy as np

class MassProfile(object):
    """
    class to deal with arbitrary mass profiles
    """

    def __init__(self, kwargs):
        self._profile_type = kwargs['type']
        self._kwargs = kwargs

    def m_r(self, r):
        """
        returns mass enclosed < r of the mass profile
        :param r:
        :return:
        """
        if self._profile_type == 'power_law':
            return self.m_r_power_law(r)
        else:
            return 0

    def m_r_power_law(self, r):
        """

        :param r:
        :return:
        """
        theta_E = self._kwargs['theta_E']
        gamma = self._kwargs['gamma']
        Sigma_crit = 1
        D_d = 1
        rho_0 = Sigma_crit * theta_E**(gamma-1) * D_d**(gamma-1) * math.gamma(gamma/2.) / (np.sqrt(np.pi) * math.gamma((gamma-3)/2.))
        return rho_0 / r**gamma