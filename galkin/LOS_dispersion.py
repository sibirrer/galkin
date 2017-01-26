__author__ = 'sibirrer'

import numpy as np
import galkin.velocity_util as vel_util
import astrofunc.constants as const


class Velocity_dispersion_numerical_integra(object):
    """
    line of sight velocity dispersion calculation
    """
    def __init__(self):
        pass

    def sigma_r2(self, r, a, gamma, rho0_r0_gamma, r_ani):
        """
        equation (19) in Suyu+ 2010
        """
        # first term
        prefac1 = 4*np.pi * const.G * a**(-gamma) * rho0_r0_gamma / (3-gamma)
        prefac2 = r * (r + a)**3/(r**2 + r_ani**2)
        hyp1 = vel_util.hyp_2F1(a=2+gamma, b=gamma, c=3+gamma, z=1./(1+r/a))
        hyp2 = vel_util.hyp_2F1(a=3, b=gamma, c=1+gamma, z=-a/r)
        fac = r_ani**2/a**2 * hyp1 / ((2+gamma) * (r/a + 1)**(2+gamma)) + hyp2 / (gamma*(r/a)**gamma)
        return prefac1 * prefac2 * fac

    def I_H_sigma(self, R, a, gamma, rho0_r0_gamma, r_ani, num_log=10, r_min=10**(-6)):
        """
        luminosity-weighted projected velocity dispersion sigma_s as a function of projected radius R
        equation (21) in Suyu+ 2010
        :return:
        """
        r_array = np.logspace(np.log10(r_min), np.log10(np.sqrt(10**2-R**2)), int((1.-np.log10(R+r_min))*num_log))
        result = 0
        for i in range(1, len(r_array)):
            r = np.sqrt(((r_array[i-1]+r_array[i])/2)**2+R**2)
            delta = r_array[i] - r_array[i-1]
            result += self._sigma_integrand(r, R, a, gamma, rho0_r0_gamma, r_ani)*delta
        #result, error = quad(self._sigma_integrand, R+0.00001, 20, args=(R, a, gamma, rho0_r0_gamma, r_ani))
        return result

    def I_H(self, R, a, I0=1, num_log=10, r_min=10**(-6)):
        """
        luminosity-weighting as a function of projected radius R
        equation (21) in Suyu+ 2010
        :return:
        """
        r_array = np.logspace(np.log10(r_min), np.log10(np.sqrt(10**2-R**2)), int((1.-np.log10(R+r_min))*num_log))
        result = 0
        for i in range(1, len(r_array)):
            r = np.sqrt(((r_array[i-1]+r_array[i])/2)**2+R**2)
            delta = r_array[i] - r_array[i-1]
            result += self._Hernquist_integrand(r, a, I0)*delta
        #result, error = quad(self._Hernquist_integrand, R+0.00001, 20, args=(a, R, I0))
        return result

    def _sigma_integrand(self, r, R, a, gamma, rho0_r0_gamma, r_ani, I0=1):
        """

        :return:
        """
        return (1 - self._beta_ani(r, r_ani) * R**2/r**2) * self._rho_star(r, a, I0) * self.sigma_r2(r, a, gamma, rho0_r0_gamma, r_ani)

    def _Hernquist_integrand(self, r, a, I0=1):
        # normalization of the Hernquist profile (not needed, cancels out)

        return self._rho_star(r, a, I0)

    def _beta_ani(self, r, r_ani):
        return r**2/(r_ani**2 + r**2)

    def _rho_star(self, r, a, I0=1):
        return I0 * a / (2*np.pi * r * (r + a)**3)

    def _Ip_R(self, R, a, I0=1):
        """
        equation 4.4 in Lee 2015 et al.
        http://arxiv.org/pdf/1410.7770v2.pdf
        computes the projected Hernquist profile on projected radius R and a = 0.551 r_eff
        :param R:
        :param a:
        :return:
        """
        s = R/a
        return I0/(2*np.pi * a**2* (1-s**2)**2) *((2+s**2)*self._X(s) - 3)

    def _X(self, s):
        """
        equation 4.5 in Lee 2015 et al
        :param s:  R/a
        :return:
        """
        if 0 <= s and s <= 1:
            X = 1/np.sqrt(1-s**2)*np.arccosh(1/s)
        elif 1 < s:
            X = 1/np.sqrt(s**2-1)*np.arccosh(1/s)
        else:
            X = 0
        return 0


class Velocity_dispersion(object):
    """
    class to compute eqn 20 in Suyu+2010 with a monte-carlo process
    """
    def __init__(self, beta_const=False, b_prior=False):
        self.beta_const = beta_const
        self.b_prior = b_prior

    def anisotropy_set_up(self, beta_const=False, b_prior=False):
        """

        :param beta_const:
        :return:
        """
        self.beta_const = beta_const
        self.b_prior = b_prior

    def vel_disp(self, gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM, num=100):
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
        if self.b_prior and self.beta_const:
            aniso_param = self.b_beta(aniso_param)
        sigma_s2_sum = 0
        for i in range(0, num):
            sigma_s2_draw = self.vel_disp_one(gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM)
            sigma_s2_sum += sigma_s2_draw
        sigma_s2_average = sigma_s2_sum/num
        return sigma_s2_average

    def vel_disp_one(self, gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM):
        """
        computes one realisation of the velocity dispersion realized in the slit
        :param gamma:
        :param rho0_r0_gamma:
        :param r_eff:
        :param r_ani:
        :param R_slit:
        :param dR_slit:
        :param FWHM:
        :return:
        """
        a = 0.551 * r_eff
        while True:
            r = self.P_r(a)  # draw r
            R, x, y = self.R_r(r)  # draw projected R
            x_, y_ = self.displace_PSF(x, y, FWHM)  # displace via PSF
            bool = self.check_in_slit(x_, y_, R_slit, dR_slit)
            if bool is True:
                break
        sigma_s2 = self.sigma_s2(r, R, aniso_param, a, gamma, rho0_r0_gamma)
        return sigma_s2

    def P_r(self, a):
        """

        :param a: 0.551*r_eff
        :return: realisation of radius of Hernquist luminosity weighting in 3d
        """
        P = np.random.uniform()  # draws uniform between [0,1)
        r = a*np.sqrt(P)*(np.sqrt(P)+1)/(1-P)  # solves analytically to r from P(r)
        return r

    def R_r(self, r):
        """
        draws a random projection from radius r in 2d and 1d
        :param r: 3d radius
        :return: R, x, y
        """
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        R = np.sqrt(x**2 + y**2)
        return R, x, y

    def displace_PSF(self, x, y, FWHM):
        """

        :param x: x-coord (arc sec)
        :param y: y-coord (arc sec)
        :param FWHM: psf size (arc sec)
        :return: x', y' random displaced according to psf
        """
        sigma = FWHM/(2*np.sqrt(2*np.log(2)))
        sigma_one_direction = sigma/np.sqrt(2)
        x_ = x + np.random.normal() * sigma_one_direction
        y_ = y + np.random.normal() * sigma_one_direction
        return x_, y_

    def check_in_slit(self, x_, y_, R_slit, dR_slit):
        """
        check whether a ray in position (x_,y_) is captured in the slit with Radius R_slit and width dR_slit
        :param x_:
        :param y_:
        :param R_slit:
        :param dR_slit:
        :return:
        """
        if abs(x_) < R_slit/2. and abs(y_) < dR_slit/2.:
            return True
        else:
            return False

    def sigma_s2(self, r, R, aniso_param, a, gamma, rho0_r0_gamma):
        """
        projected velocity dispersion
        :param r:
        :param R:
        :param r_ani:
        :param a:
        :param gamma:
        :param phi_E:
        :return:
        """
        if self.beta_const:
            beta = aniso_param
            r_ani = self._ani_beta(r, aniso_param)
        else:
            r_ani = aniso_param
            beta = self._beta_ani(r, r_ani)
        return (1 - beta * R**2/r**2) * self.sigma_r2(r, a, gamma, rho0_r0_gamma, r_ani)

    def sigma_r2(self, r, a, gamma, rho0_r0_gamma, r_ani):
        """
        equation (19) in Suyu+ 2010
        """
        # first term
        prefac1 = 4*np.pi * const.G * a**(-gamma) * rho0_r0_gamma / (3-gamma)
        prefac2 = r * (r + a)**3/(r**2 + r_ani**2)
        hyp1 = vel_util.hyp_2F1(a=2+gamma, b=gamma, c=3+gamma, z=1./(1+r/a))
        hyp2 = vel_util.hyp_2F1(a=3, b=gamma, c=1+gamma, z=-a/r)
        fac = r_ani**2/a**2 * hyp1 / ((2+gamma) * (r/a + 1)**(2+gamma)) + hyp2 / (gamma*(r/a)**gamma)
        return prefac1 * prefac2 * fac

    def _beta_ani(self, r, r_ani):
        """
        anisotropy parameter beta
        :param r:
        :param r_ani:
        :return:
        """
        #return 0
        return r**2/(r_ani**2 + r**2)

    def _ani_beta(self, r, beta):
        """
        given radius and anisotropy beta, what is the "anisotropy radius"
        :param r:
        :param beta:
        :return:
        """
        if beta > 1:
            raise ValueError("Value of beta = %s not valid!" % beta)
        return np.sqrt(r**2*(1./beta-1))

    def b_beta(self, b):
        """

        :param b: 1 - 1/b = beta
        :return:
        """
        assert( b>0 )
        return 1. -1./b