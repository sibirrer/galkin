"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
import numpy as np
from galkin.light_profile import LightProfile
from lenstronomy.Analysis.lens_analysis import LensAnalysis


class TestLightProfile(object):

    def setup(self):
        pass

    def test_draw_light(self):
        lightProfile = LightProfile(profile_list=['HERNQUIST'])
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.8}]
        r_list = lightProfile.draw_light_2d(kwargs_profile, n=100000)
        bins = np.linspace(0., 1, 20)
        hist, bins_hist = np.histogram(r_list, bins=bins, normed=True)
        light2d = lightProfile.light_2d(R=(bins_hist[1:] + bins_hist[:-1])/2., kwargs_list=kwargs_profile)
        light2d *= (bins_hist[1:] + bins_hist[:-1]) / 2.
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        print light2d / hist
        for i in range(len(hist)):
            print bins_hist[i], i, light2d[i] / hist[i]
            npt.assert_almost_equal(light2d[i] / hist[i], 1, decimal=1)

    def test_draw_light_PJaffe(self):
        lightProfile = LightProfile(profile_list=['PJAFFE'])
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.5, 'Ra': 0.2}]
        r_list = lightProfile.draw_light_2d(kwargs_profile, n=50000)
        bins = np.linspace(0, 1, 10)
        hist, bins_hist = np.histogram(r_list, bins=bins, normed=True)
        light2d = lightProfile.light_2d(R=(bins_hist[1:] + bins_hist[:-1])/2., kwargs_list=kwargs_profile)
        light2d *= (bins_hist[1:] + bins_hist[:-1]) / 2.
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        #print light2d / hist
        npt.assert_almost_equal(light2d[4]/hist[4], 1, decimal=2)

        lightProfile = LightProfile(profile_list=['PJAFFE'], kwargs_numerics={'min_interpolate': 0.0001, 'max_interpolate': 20.})
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.04, 'Ra': 0.02}]
        r_list = lightProfile.draw_light_2d(kwargs_profile, n=50000)
        bins = np.linspace(0., 0.1, 10)
        hist, bins_hist = np.histogram(r_list, bins=bins, normed=True)
        light2d = lightProfile.light_2d(R=(bins_hist[1:] + bins_hist[:-1])/2., kwargs_list=kwargs_profile)
        light2d *= (bins_hist[1:] + bins_hist[:-1]) / 2.
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        print light2d / hist
        npt.assert_almost_equal(light2d[4] / hist[4], 1, decimal=2)

    def test_ellipticity_in_profiles(self):
        lightProfile = ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': 0.020568531548241405,
            'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEMD'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': lightProfile}
        lensAnalysis = LensAnalysis(kwargs_options, {})
        r_eff = lensAnalysis.half_light_radius(kwargs_profile)
        kwargs_profile[0]['q'] = 1
        kwargs_profile[1]['q'] = 1
        r_eff_spherical = lensAnalysis.half_light_radius(kwargs_profile)
        npt.assert_almost_equal(r_eff / r_eff_spherical, 1, decimal=2)

    def test_light_3d(self):
        lightProfile = LightProfile(profile_list=['HERNQUIST'])
        r = np.logspace(-2, 2, 100)
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.5}]
        light_3d = lightProfile.light_3d_interp(r, kwargs_profile)
        light_3d_exact = lightProfile.light_3d(r, kwargs_profile)
        for i in range(len(r)):
            npt.assert_almost_equal(light_3d[i]/light_3d_exact[i], 1, decimal=3)


if __name__ == '__main__':
    pytest.main()