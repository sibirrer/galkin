import numpy as np
import math
import velocity_util as vel_util
import astrofunc.constants as const

class Jeans_solver(object):
    """
    class to solve radial Jeans equation for different configuration
    """
    def __init__(self, kwargs_cosmo, mass_profile, light_profile, anisotropy_type):
        self.D_d = kwargs_cosmo['D_d']
        self.D_s = kwargs_cosmo['D_s']
        self.D_ds = kwargs_cosmo['D_ds']
        self._mass_profile = mass_profile
        self._light_profile = light_profile
        self._anisotropy_type = anisotropy_type

    def numerical_solver(self, r):
        """

        :param r:
        :return:
        """
        return 0

    def power_law_anisotropy(self, r, kwargs_profile, kwargs_anisotropy, kwargs_light):
        """
        equation (19) in Suyu+ 2010
        :param r:
        :return:
        """
        # first term
        theta_E = kwargs_profile['theta_E']
        gamma = kwargs_profile['gamma']
        r_ani = kwargs_anisotropy['r_ani']
        a = 0.551 * kwargs_light['r_eff']
        rho0_r0_gamma = self._rho0_r0_gamma(theta_E, gamma)
        prefac1 = 4*np.pi * const.G * a**(-gamma) * rho0_r0_gamma / (3-gamma)
        prefac2 = r * (r + a)**3/(r**2 + r_ani**2)
        hyp1 = vel_util.hyp_2F1(a=2+gamma, b=gamma, c=3+gamma, z=1./(1+r/a))
        hyp2 = vel_util.hyp_2F1(a=3, b=gamma, c=1+gamma, z=-a/r)
        fac = r_ani**2/a**2 * hyp1 / ((2+gamma) * (r/a + 1)**(2+gamma)) + hyp2 / (gamma*(r/a)**gamma)
        return prefac1 * prefac2 * fac


    def _rho0_r0_gamma(self, theta_E, gamma):
        # equation (14) in Suyu+ 2010
        return -1 * math.gamma(gamma/2.)/(np.sqrt(np.pi)*math.gamma((gamma-3)/2.)) * theta_E**gamma/self.arcsec2phys_lens(theta_E) * self.epsilon_crit * const.M_sun/const.Mpc**3  # units kg/m^3

    def sigma_r2(self, r, kwargs_profile, kwargs_anisotropy, kwargs_light):
        """
        solves radial Jeans equation
        """
        if self._mass_profile == 'power_law':
            if self._anisotropy_type == 'r_ani':
                if self._light_profile == 'Hernquist':
                    sigma_r = self.power_law_anisotropy(r, kwargs_profile, kwargs_anisotropy, kwargs_light)
                else:
                    raise ValueError(' light profile %s not supported for Jeans solver' % self._light_profile)
            else:
                raise ValueError('anisotropy type %s not implemented in Jeans equation modelling' % self._anisotropy_type)
        else:
            raise ValueError('mass profile type %s not implemented in Jeans solver' % self._mass_profile)
        return sigma_r

    def arcsec2phys_lens(self, theta):
        """
        converts are seconds to physical units on the deflector
        :param theta:
        :return:
        """
        return theta * const.arcsec * self.D_d

    @property
    def epsilon_crit(self):
        """
        returns the critical projected mass density in units of M_sun/Mpc^2 (physical units)
        """
        const_SI = const.c**2 / (4*np.pi * const.G)  #c^2/(4*pi*G) in units of [kg/m]
        conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
        const = const_SI*conversion   #c^2/(4*pi*G) in units of [M_sun/Mpc]
        Epsilon_Crit = self.D_s/(self.D_d*self.D_ds) * const #[M_sun/Mpc^2]
        return Epsilon_Crit