import numpy as np
import galkin.velocity_util as vel_util
import astrofunc.constants as const


class Velocity_dispersion_numerical_integral(object):
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