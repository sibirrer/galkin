"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
from galkin.galkin_old import GalKin_old
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
        gamma = 2.2
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
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_aperture, kwargs_light, kwargs_anisotropy, num=1000)

        los_disp = Velocity_dispersion(beta_const=False, b_prior=False, kwargs_cosmo=kwargs_cosmo)
        sigma_v2 = los_disp.vel_disp(gamma, theta_E, r_eff, aniso_param=r_ani, R_slit=length/2., dR_slit=width/2.,
                                     FWHM=psf_fwhm, num=1000)
        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=1)

if __name__ == '__main__':
    pytest.main()