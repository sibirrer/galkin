__author__ = 'sibirrer'

import scipy.ndimage as ndimage
import numpy as np

from galkin.LOS_dispersion import Velocity_dispersion

class Apperature(object):
    """
    this class is aimed to simulate slit and psf effecs of ground based spectrographs
    """
    def __init__(self):
        self.vel_dispersion = Velocity_dispersion()


    def get_slit_point(self, R_slit, phi_slit, center_x, center_y, psf_fwhm, num_evaluate):
        """

        :param R_slit: Slit lenght [arc sec]
        :param phi_slit: angle of slit [radian]
        :param center_x: position of center of slit in x-axis
        :param center_y: position of center of slit in y-axis
        :param psf_fwhm: FWHM of (Gaussian) PSF
        :param num_evaluate: number of points to be evaluated
        :return:
        """
        num_x = num_evaluate
        delta = R_slit/num_evaluate
        num_y = round(2*psf_fwhm/delta+0.49)*2 + 1
        x_array = np.linspace(-(num_x-1)/2.*delta, (num_x-1)/2.*delta, num_x)
        y_array = np.linspace(-(num_y-1)/2.*delta, (num_y-1)/2.*delta, int(num_y))
        grid_x, grid_y = np.meshgrid(x_array, y_array)
        mask = np.zeros_like(grid_x)
        mask[np.where(grid_y == 0)] = 1
        return grid_x, grid_y, mask

    def convolve_signal(self, grid, psf, delta=1):
        signal_conv = ndimage.filters.gaussian_filter(grid, psf/delta, mode='constant', cval=0.0, truncate=3)
        return signal_conv

    def cut_apperature(self, signal_grid, mask):
        return np.sum(signal_grid*mask)

    def LOS_velocity_dispersion_grid(self, grid_x, grid_y, r_eff, gamma, rho0_r0_gamma, r_ani, num_log, r_min):
        IH_sigma_s2_grid = np.zeros_like(grid_x)
        IH_grid = np.zeros_like(grid_x)
        R = np.sqrt(grid_x**2 + grid_y**2)
        a = 0.551 * r_eff
        for i in range(0, len(grid_x)):
            for j in range(0, len(grid_x[0])):
                IH_grid[i][j] = self.vel_dispersion.I_H(R[i][j], a, num_log=num_log, r_min=r_min)
                IH_sigma_s2_grid[i][j] = self.vel_dispersion.I_H_sigma(R[i][j], a, gamma, rho0_r0_gamma, r_ani, num_log=num_log, r_min=r_min)
        return IH_grid, IH_sigma_s2_grid

    def LOS_velocity_dispersion_measure(self, r_eff, gamma, rho0_r0_gamma, r_ani, R_slit, phi_slit=0, center_x=0, center_y=0, psf_fwhm=0.7, num_evaluate=11, num_log=10, r_min=10**(-6)):
        grid_x, grid_y, mask = self.get_slit_point(R_slit, phi_slit, center_x, center_y, psf_fwhm, num_evaluate)
        IH_grid, IH_sigma_s2_grid = self.LOS_velocity_dispersion_grid(grid_x, grid_y, r_eff, gamma, rho0_r0_gamma, r_ani, num_log, r_min)
        #IH_grid_convolved = IH_grid
        #IH_grid_sigma_s2_convolved = IH_sigma_s2_grid
        delta = R_slit/num_evaluate
        IH_grid_convolved = self.convolve_signal(IH_grid, psf_fwhm, delta)
        IH_grid_sigma_s2_convolved = self.convolve_signal(IH_sigma_s2_grid, psf_fwhm, delta)
        return self.cut_apperature(IH_grid_sigma_s2_convolved, mask)/self.cut_apperature(IH_grid_convolved, mask)