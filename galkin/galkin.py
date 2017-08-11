from light_profile import LightProfile
from mass_profile import MassProfile
from aperture import Aperture
from anisotropy import MamonLokasAnisotropy
from cosmo import Cosmo
import velocity_util as util
import astrofunc.constants as const

import numpy as np


class Galkin(object):
    """
    major class to compute velocity dispersion measurements given light and mass models
    """
    def __init__(self, mass_profile_list, light_profile_list, aperture_type='slit', anisotropy_model='isotropic',
                 fwhm=0.7, kwargs_cosmo={'D_d': 1000, 'D_s': 2000, 'D_ds': 500}):
        self.massProfile = MassProfile(mass_profile_list, kwargs_cosmo)
        self.lightProfile = LightProfile(light_profile_list)
        self.aperture = Aperture(aperture_type)
        self.anisotropy = MamonLokasAnisotropy(anisotropy_model)
        self.FWHM = fwhm
        self.cosmo = Cosmo(kwargs_cosmo)

    def vel_disp(self, kwargs_mass, kwargs_light, kwargs_anisotropy, kwargs_apertur, num=100):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)
        :param gamma:
        :param phi_E:
        :param r_eff:
        :param r_ani:
        :param R_slit:
        :param FWHM:
        :return:
        """
        sigma2_R_sum = 0
        for i in range(0, num):
            sigma2_R = self.draw_one_sigma2(kwargs_mass, kwargs_light, kwargs_anisotropy, kwargs_apertur)
            sigma2_R_sum += sigma2_R
        sigma_s2_average = sigma2_R_sum / num
        # apply unit conversion from arc seconds and deflections to physical velocity disperison in (km/s)
        sigma_s2_average *= 2 * const.G  # correcting for integral prefactor
        return np.sqrt(sigma_s2_average/(const.arcsec**2 * self.cosmo.D_d**2 * const.Mpc))/1000.  # in units of km/s

    def draw_one_sigma2(self, kwargs_mass, kwargs_light, kwargs_anisotropy, kwargs_aperture):
        """

        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :param kwargs_aperture:
        :return:
        """
        while True:
            R = self.lightProfile.draw_light_2d(kwargs_light)  # draw r
            x, y = util.draw_xy(R)  # draw projected R
            x_, y_ = util.displace_PSF(x, y, self.FWHM)  # displace via PSF
            bool = self.aperture.aperture_select(x_, y_, kwargs_aperture)
            if bool is True:
                break
        sigma2_R = self.sigma2_R(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        return sigma2_R

    def sigma2_R(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion
        :param R:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
        """
        I_R_sigma2 = self.I_R_simga2(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        I_R = self.lightProfile.light_2d(R, kwargs_light)
        return I_R_sigma2 / I_R

    def I_R_simga2(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy, num=100):
        """
        equation A15 in Mamon&Lokas 2005 as a logarithmic numerical integral
        modulo pre-factor 2*G
        :param R:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
        """
        IR_sigma2 = 0
        dr_array = np.linspace(R+0.000001, 20, num)
        dr = dr_array[1] - dr_array[0]
        for r in dr_array:
            IR_sigma2 += self._integrand_A15(r, R, kwargs_mass, kwargs_light, kwargs_anisotropy) * dr
        return IR_sigma2

    def _integrand_A15(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A15 (in log space)
        :param r:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
        """
        k_r = self.anisotropy.K(r, R, kwargs_anisotropy)
        l_r = self.lightProfile.light_3d(r, kwargs_light)
        m_r = self.massProfile.mass_3d(r, kwargs_mass)
        return k_r * l_r * m_r / r
