import numpy as np
import copy
from scipy.interpolate import interp1d
from lenstronomy.ImSim.light_model import LightModel


class LightProfile(object):
    """
    class to deal with the light distribution
    """
    def __init__(self, profile_list=['HERNQUIST']):
        """

        :param profile_list:
        """
        self.light_model = LightModel(profile_type_list=profile_list)

    def light_3d(self, r, kwargs_list):
        """

        :param kwargs_list:
        :return:
        """
        return self.light_model.light_3d(r, kwargs_list)

    def light_2d(self, R, kwargs_list):
        """

        :param R:
        :param kwargs_list:
        :return:
        """
        return self.light_model.surface_brightness(R, 0, kwargs_list)

    def draw_light_2d(self, kwargs_list, n=1, new_compute=False):
        """
        constructs the CDF and draws from it random realizations of projected radii R
        :param kwargs_list:
        :return:
        """
        if not hasattr(self, '_light_cdf') or new_compute is True:
            r_array = np.linspace(0,10, 500)
            cum_sum = np.zeros_like(r_array)
            sum = 0
            for i, r in enumerate(r_array):
                sum += self.light_2d(r, kwargs_list) * r
                cum_sum[i] = copy.deepcopy(sum)
            cum_sum_norm = cum_sum/cum_sum[-1]
            f = interp1d(cum_sum_norm, r_array)
            self._light_cdf = f
        cdf_draw = np.random.uniform(0, 1, n)
        r_draw = self._light_cdf(cdf_draw)
        return r_draw


class LightProfile_old(object):
    """
    class to deal with the light distribution
    """
    def __init__(self, profile_type='Hernquist'):
        self._profile_type = profile_type

    def draw_light(self, kwargs_light):
        """

        :param kwargs_light:
        :return:
        """
        if self._profile_type == 'Hernquist':
            r = self.P_r_hernquist(kwargs_light)
        else:
            raise ValueError('light profile %s not supported!')
        return r

    def P_r_hernquist(self, kwargs_light):
        """

        :param a: 0.551*r_eff
        :return: realisation of radius of Hernquist luminosity weighting in 3d
        """
        r_eff = kwargs_light['r_eff']
        a = 0.551 * r_eff
        P = np.random.uniform()  # draws uniform between [0,1)
        r = a*np.sqrt(P)*(np.sqrt(P)+1)/(1-P)  # solves analytically to r from P(r)
        return r
