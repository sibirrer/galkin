import numpy as np


class Anisotorpy(object):
    """
    class that handels the kinematic anisotropy
    """
    def __init__(self, anisotropy_type):
        self._type = anisotropy_type

    def beta_r(self, r, kwargs):
        """
        returns the anisotorpy parameter at a given radius
        :param r:
        :return:
        """
        if self._type == 'const':
            return self.const_beta(kwargs)
        elif self._type == 'r_ani':
            return self.beta_r_ani(r, kwargs)
        else:
            raise ValueError('anisotropy type %s not supported!' % self._type)

    def J_beta_rs(self, r, s, kwargs):
        """

        :param r:
        :param s:
        :return:
        """
        if r <= 0:
            r = 0.00000001
        if self._type == 'r_ani':
            r_ani = kwargs['r_ani']
            beta_infty = kwargs.get('beta_infty', 1)
            return ((s**2 + r_ani**2) / (r**2 + r_ani**2))**beta_infty
        elif self._type == 'const':
            beta = kwargs['beta']
            return (s / r)**(2*beta)
        else:
            n = 100
            r_ = np.linspace(r, s, n)
            int = 0
            for r_i in r_:
                int += 2 * self.beta_r(r_i, kwargs)/r_i
            int *= (r-s) / n
            return np.exp(int)

    def const_beta(self, kwargs):
        return kwargs['beta']

    def beta_r_ani(self, r, kwargs):
        """

        :param r:
        :return:
        """
        return self._beta_ani(r, kwargs['r_ani'])

    def _beta_ani(self, r, r_ani):
        """
        anisotropy parameter beta
        :param r:
        :param r_ani:
        :return:
        """
        return r**2/(r_ani**2 + r**2)