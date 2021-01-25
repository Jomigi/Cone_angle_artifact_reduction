#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:37:00 2019

@author: Jordi Minnema, Felix Lucka
"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import imageio
import tifffile
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp2d
from glob import glob
from natsort import natsorted
import skimage.measure as measure


class Radial2Axial(object):
    def __init__(self, phantom, architecture, depth, pos, it, base_path, run_folder):
        """
        :param xs:
        :param ys:
        :param batch_size:
        """
        self.phantom = phantom
        self.architecture = architecture
        self.it = it
        self.depth = depth
        self.pos = pos
        self.base_path = base_path
        self.run_folder = run_folder

    def radial2axial(self, rad_slices):
        n_rad = rad_slices.shape[0]
        n_x = rad_slices.shape[1]
        n_z = rad_slices.shape[2]

        # We want to describe the voxels of the radial slices in polar coordinates. Therefore, we define a theta to describe the angles of the radial slices.
        theta = np.linspace(0, np.pi, n_rad+1, endpoint=True)
        print(rad_slices.shape)
        # we add the first radial slice (theta = 0) to the end to be the data for theta = pi
        rad_slices = np.concatenate((rad_slices,(rad_slices[0,:,:]).reshape(1,n_x,n_x)),axis=0)
        x_gr = np.linspace(-1.0, 1.0, n_x)

        # set up radial and angular coordinates of the x-y grid (needs to be done once for all z slices)
        polar_cords = np.zeros((n_x**2, 2))

        for i in range(n_x*n_x):

            # x-coordinate
            x = x_gr[np.mod(i, n_x)]

            # y-coordinate
            y = x_gr[i//n_x]

            if x == 0:
                polar_cords[i, 0] = np.pi/2  # angular
                polar_cords[i, 1] = y  # radial

            if y == 0:
                polar_cords[i, 0] = 0 # angular
                polar_cords[i, 1] = x # radial

            elif x != 0 and y != 0:
                polar_cords[i, 0] = np.arctan(y/x)   # angular
                polar_cords[i, 1] = np.sign(y)*np.sqrt(x**2 + y**2)  # radial

            if polar_cords[i,0] < 0:
                polar_cords[i,0] =  polar_cords[i,0] + np.pi # Keep angle positive

        axial_slices = np.zeros([n_z, n_x, n_x])

        for z in range(n_z):
            interpolator = rgi((theta, x_gr), rad_slices[:,z,:], method='linear', bounds_error=False, fill_value=0)
            axial_slices[z,:,:] = interpolator(polar_cords).reshape(1,n_x,n_x, order='F')

        return axial_slices[:,:,::-1]

    def generate_rad_2_ax(self):
        for pos_i in self.pos:
            for ii in self.it:
                for depth_i in self.depth:

                    path_to_radial= self.base_path + 'ConeBeamResults/'+ self.run_folder +'{}_pos{}_it{}_depth{}_phantom{}/*'.format(self.architecture, pos_i, ii, depth_i,self.phantom)
                    #path_to_radial= '/export/scratch2/home/shannon/ConeBeamResults/predictions/msd_pos1_it1_depth80_walnut9/*'  #so this is still for each walnut individualy not the whole folder
                    radial_slices = natsorted(glob(path_to_radial))
                    print(path_to_radial)
                    nb_slices = len(radial_slices)
                    img_h = 501
                    img_w = 501
                    print(nb_slices)
                    rad_image = np.zeros([nb_slices, img_h, img_w])

                    for i,rslice in enumerate(radial_slices):
                        rad_image[i,:,:] = imageio.imread(rslice)

                    radial2axial_image = self.radial2axial(rad_image)

                    for k in range(radial2axial_image.shape[0]):
                        save_path = self.base_path + 'ReconResults/'+ self.run_folder + '{}_pos{}_it{}_depth{}_phantom{}/'.format(self.architecture, pos_i, ii, depth_i,self.phantom)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        tifffile.imsave(save_path + 'slice_{foo:03d}.tif'.format(foo=k), radial2axial_image[k,:,:])

