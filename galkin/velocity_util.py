__author__ = 'sibirrer'

import mpmath as mp
import numpy as np


def hyp_2F1(a, b, c, z):
    """
    http://docs.sympy.org/0.7.1/modules/mpmath/functions/hypergeometric.html
    """
    return mp.hyp2f1(a, b, c, z)


def displace_PSF(x, y, FWHM):
    """

    :param x: x-coord (arc sec)
    :param y: y-coord (arc sec)
    :param FWHM: psf size (arc sec)
    :return: x', y' random displaced according to psf
    """
    sigma = FWHM/ (2 * np.sqrt(2 * np.log(2)))
    sigma_one_direction = sigma / np.sqrt(2)
    x_ = x + np.random.normal() * sigma_one_direction
    y_ = y + np.random.normal() * sigma_one_direction
    return x_, y_