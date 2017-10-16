"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
import numpy as np
import scipy.integrate as integrate
from galkin.galkin_old import GalKin_old
from galkin.galkin import Galkin
from galkin.light_profile import LightProfile
from galkin.LOS_dispersion import Velocity_dispersion


class TestGalkin(object):

    def setup(self):
        pass

    def test_test(self):
        assert 0 == 0

    def test_galkin_vs_LOS_dispersion(self):
        """
        tests whether the old and new version provide the same answer
        :return:
        """
        # light profile
        light_profile = 'Hernquist'
        r_eff = 0.5
        kwargs_light = {'r_eff': r_eff}  # effective half light radius (2d projected) in arcsec

        # mass profile
        mass_profile = 'power_law'
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = {'theta_E': theta_E, 'gamma': gamma}  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'r_ani'
        r_ani = 0.5
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as shell
        #aperture_type = 'shell'
        #kwargs_aperture_inner = {'r_in': 0., 'r_out': 0.2, 'center_dec': 0, 'center_ra': 0}

        #kwargs_aperture_outer = {'r_in': 0., 'r_out': 1.5, 'center_dec': 0, 'center_ra': 0}

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'D_d': 1000, 'D_s': 1500, 'D_ds': 800}
        galkin = GalKin_old(aperture=aperture_type, mass_profile=mass_profile, light_profile=light_profile,
                            anisotropy_type=anisotropy_type, psf_fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo)
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_aperture, kwargs_light, kwargs_anisotropy, num=100)

        los_disp = Velocity_dispersion(beta_const=False, b_prior=False, kwargs_cosmo=kwargs_cosmo)
        sigma_v2 = los_disp.vel_disp(gamma, theta_E, r_eff, aniso_param=r_ani, R_slit=length/2., dR_slit=width/2.,
                                     FWHM=psf_fwhm, num=100)
        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=1)

    def test_log_vs_linear_integral(self):
        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = 1.8
        kwargs_light = [{'Rs':  r_eff, 'sigma0': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OsipkovMerritt'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'D_d': 1000, 'D_s': 1500, 'D_ds': 800}
        galkin = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type, anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo)
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture, num=1000)
        sigma_v2 = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture, num=1000, log_int=True)
        print sigma_v, sigma_v2, 'sigma_v linear, sigma_v log'
        print (sigma_v/sigma_v2)**2

        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=1)

    def test_compare_power_law(self):
        """
        compare power-law profiles analytical vs. numerical
        :return:
        """
        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = 1.8
        kwargs_light = [{'Rs':  r_eff, 'sigma0': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OsipkovMerritt'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'D_d': 1000, 'D_s': 1500, 'D_ds': 800}
        galkin = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type, anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo)
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture, num=1000, log_int=True)

        los_disp = Velocity_dispersion(beta_const=False, b_prior=False, kwargs_cosmo=kwargs_cosmo)
        sigma_v2 = los_disp.vel_disp(gamma, theta_E, r_eff, aniso_param=r_ani, R_slit=length/2., dR_slit=width/2.,
                                     FWHM=psf_fwhm, num=1000)
        print sigma_v, sigma_v2, 'sigma_v Galkin, sigma_v los dispersion'
        print (sigma_v/sigma_v2)**2

        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=1)

    def test_numeric_light_integral(self):
        """

        :return:
        """
        light_profile_list = ['HERNQUIST']
        r_eff = .05
        kwargs_light = [{'Rs': r_eff, 'sigma0': 1.}]  # effective half light radius (2d projected) in arcsec
        lightProfile = LightProfile(light_profile_list)
        R = 2
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        light2d_int = lightProfile._integrand_light(R, kwargs_light)
        npt.assert_almost_equal(light2d/light2d_int, 1, decimal=2)

    def test_projected_light_integral_hernquist(self):
        """

        :return:
        """
        light_profile_list = ['HERNQUIST']
        r_eff = 1.
        kwargs_light = [{'Rs': r_eff, 'sigma0': 1.}]  # effective half light radius (2d projected) in arcsec
        lightProfile = LightProfile(light_profile_list)
        R = 2
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 10)
        npt.assert_almost_equal(light2d, out[0]*2, decimal=3)

    def test_projected_light_integral_hernquist_ellipse(self):
        """

        :return:
        """
        light_profile_list = ['HERNQUIST_ELLIPSE']
        r_eff = 1.
        kwargs_light = [{'Rs': r_eff, 'sigma0': 1., 'q': 0.8, 'phi_G': 1.}]  # effective half light radius (2d projected) in arcsec
        lightProfile = LightProfile(light_profile_list)
        R = 2
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 10)
        npt.assert_almost_equal(light2d, out[0]*2, decimal=3)

    def test_projected_light_integral_pjaffe(self):
        """

        :return:
        """
        light_profile_list = ['PJAFFE']
        kwargs_light = [{'Rs': .5, 'Ra': 0.01, 'sigma0': 1.}]  # effective half light radius (2d projected) in arcsec
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print out, 'out'
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)

    def test_realistic_0(self):
        """
        realistic test example
        :return:
        """
        light_profile_list = ['HERNQUIST']
        kwargs_light = [{'Rs': 0.10535462602138289, 'center_x': -0.02678473951679429, 'center_y': 0.88691126347462712, 'sigma0': 3.7114695634960109}]
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print out, 'out'
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)

    def test_realistic_1(self):
        """
        realistic test example
        :return:
        """
        light_profile_list = ['HERNQUIST_ELLIPSE']
        kwargs_light = [{'Rs': 0.10535462602138289, 'q': 0.46728323131925864, 'center_x': -0.02678473951679429, 'center_y': 0.88691126347462712, 'phi_G': 0.74260706384506325, 'sigma0': 3.7114695634960109}]
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print out, 'out'
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)

    def test_realistic(self):
        """
        realistic test example
        :return:
        """
        light_profile_list = ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']
        kwargs_light = [{'Rs': 0.10535462602138289, 'q': 0.46728323131925864, 'center_x': -0.02678473951679429, 'center_y': 0.88691126347462712, 'phi_G': 0.74260706384506325, 'sigma0': 3.7114695634960109}, {'Rs': 0.44955054610388684, 'q': 0.66582356813012267, 'center_x': 0.019536801118136753, 'center_y': 0.0218888643537157, 'Ra': 0.0010000053334891974, 'phi_G': -0.33379268413794494, 'sigma0': 967.00280526319796}]
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print out, 'out'
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)


if __name__ == '__main__':
    pytest.main()