

class GalKin(object):
    """
    master class for all computations
    """
    def __init__(self, aperture_type='slit', mass_profile='power_law', psf_fwhm=0.7, kwargs_aperture={}, kwargs_profile={}):
        """
        initializes the observation condition and masks
        :param aperture_type: string
        :param psf_fwhm: float
        """
        self._aperture_type = aperture_type
        self._mass_profile = mass_profile
        self._fwhm = psf_fwhm
        self._kwargs_aperture = kwargs_aperture