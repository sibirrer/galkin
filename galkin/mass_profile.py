import math
import numpy as np
from scipy.interpolate import interp1d
from lenstronomy.ImSim.lens_model import LensModel
import astrofunc.constants as const
from cosmo import Cosmo

class MassProfile(object):
    """
    mass profile class
    """
    def __init__(self, profile_list, kwargs_cosmo={'D_d': 1000, 'D_s': 2000, 'D_ds': 500}):
        """

        :param profile_list:
        """
        kwargs_options = {'lens_model_list': profile_list}
        self.model = LensModel(kwargs_options)
        self.cosmo = Cosmo(kwargs_cosmo)

    def mass_3d_interp(self, r, kwargs, new_compute=False):
        """

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :return: mass enclosed physical radius in kg
        """
        if not hasattr(self, '_log_mass_3d') or new_compute is True:
            r_array = np.linspace(0.0001, 20, 200)
            mass_3d_array = self.model.mass_3d(r_array, kwargs)
            mass_dim_array = mass_3d_array * const.arcsec ** 3 * self.cosmo.D_d ** 2 * self.cosmo.D_s \
                       / self.cosmo.D_ds * const.Mpc * const.c ** 2 / (4 * np.pi * const.G)
            f = interp1d(r_array, np.log(mass_dim_array/r_array), fill_value="extrapolate")
            self._log_mass_3d = f
        return np.exp(self._log_mass_3d(r)) * r

    def mass_3d(self, r, kwargs):
        """

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :return: mass enclosed physical radius in kg
        """
        mass_dimless = self.model.mass_3d(r, kwargs)
        mass_dim = mass_dimless * const.arcsec ** 3 * self.cosmo.D_d ** 2 * self.cosmo.D_s \
                       / self.cosmo.D_ds * const.Mpc * const.c ** 2 / (4 * np.pi * const.G)
        return mass_dim


class MassProfile_old(object):
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