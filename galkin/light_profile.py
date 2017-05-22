import numpy as np


class LightProfile(object):
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
