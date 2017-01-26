__author__ = 'sibirrer'

import mpmath as mp

def hyp_2F1(a, b, c, z):
    """
    http://docs.sympy.org/0.7.1/modules/mpmath/functions/hypergeometric.html
    """
    return mp.hyp2f1(a, b, c, z)