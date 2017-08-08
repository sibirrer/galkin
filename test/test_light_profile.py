"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
import numpy as np
from galkin.light_profile import LightProfile


class TestLightProfile(object):

    def setup(self):
        pass

    def test_draw_light(self):
        lightProfile = LightProfile(profile_list=['HERNQUIST'])
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.5}]
        r_list = lightProfile.draw_light_2d(kwargs_profile, n=10000)
        npt.assert_almost_equal(np.median(r_list), 0.8, decimal=1)

if __name__ == '__main__':
    pytest.main()