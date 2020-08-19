#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Felix Lucka, CWI, Amsterdam
Felix.Lucka@cwi.nl

TODO

fhe full
data of one of the data sets described in
"A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning" by
Henri Der Sarkissian, Felix Lucka, Maureen van Eijnatten,
Giulia Colacicco, Sophia Bethany Coban, Kees Joost Batenburg
"""


import numpy as np
import astra
import os
import imageio
import time
import matplotlib.pyplot as plt
import nesterov_gradient
from   scipy.interpolate import RegularGridInterpolator as rgi

def rotate_astra_vec_geom(vecs, theta):

    s = np.asmatrix(vecs[:,0:3])
    d = np.asmatrix(vecs[:,3:6])
    u = np.asmatrix(vecs[:,6:9])
    v = np.asmatrix(vecs[:,9:12])

    du = d + u
    dv = d + v

    rot_mat = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [0, 0, 1]])
    s      = s * rot_mat.transpose()
    d      = d * rot_mat.transpose()
    du     = du * rot_mat.transpose()
    dv     = dv * rot_mat.transpose()

    u      = du - d
    v      = dv - d

    vecs = np.concatenate((np.asarray(s), np.asarray(d), np.asarray(u), np.asarray(v)), axis=1)

    return vecs

#### user defined settings ####################################################

# select the ID of the sample you want to reconstruct
walnut_id = 1
# define a sub-sampling factor in angular direction
# (all reference reconstructions are computed with full angular resolution)
angluar_sub_sampling = 1
# select of voxels per mm in one direction (higher = larger res)
# volume size in one direction will be 50 * voxel_per_mm + 1
voxel_per_mm = 10
# the number of slices to be extracted will be number of voxels in one direction
# times this factor
radial_slice_fac = np.sqrt(2)
# to avoid artefacts from the radial slicing, we compute multiple iterative
# reconstructions with rotated geometries and only extract the radial slices that are close
# t0 0 and 90 degrees. n_div is the number of reconstruction
n_div = 24
# we enter here some intrinsic details of the dataset needed for our reconstruction scripts
# set the variable "data_path" to the path where the dataset is stored on your own workstation
data_path = '/bigstore/felix/Walnuts/'
# set the variable "recon_path" to the path where you would like to store the
# reconstructions you compute
rad_slice_path = '/bigstore/felix/WalnutsRadialSlices/'
# set index of gpu to use
gpu_index = 3;
astra.astra.set_gpu_index(gpu_index)


print('computing Walnut', walnut_id, ',on GPU', gpu_index, flush=True)

#### general settings #########################################################

# projection index
# there are in fact 1201, but the last and first one come from the same angle
projs_idx      = range(0,1200, angluar_sub_sampling)
nb_projs_orbit = len(projs_idx)
projs_name     = 'scan_{:06}.tif'
dark_name      = 'di000000.tif'
flat_name      = ['io000000.tif', 'io000001.tif']
vecs_name      = 'scan_geom_corrected.geom'
projs_rows     = 972
projs_cols     = 768

# transformation to apply to each image, we need to get the image from
# the way the scanner reads it out into to way described in the projection
# geometry
trafo = lambda image : np.transpose(np.flipud(image))

# size of the reconstruction volume in voxels
n_x     = 50 * voxel_per_mm + 1
# size of a cubic voxel in mm
vox_sz  = 1/voxel_per_mm
# number of radial slices to be extracted
n_rad   = int(np.round(n_x * radial_slice_fac))
# angles of radial slices to be extracted
theta   = np.linspace(0,np.pi, n_rad, False)

### set up angle division
center_angles        = np.linspace(0, np.pi/2, 2*n_div+1)[1::2]
angle_div_half_width = (center_angles[1] - center_angles[0])/2 + np.pi * 10**(-16)


#### FDK reconstructions ######################################################

t_fdk = time.time();

for orbit_id in [1,2,3]: # loop over orbits

    ### load data #############################################################

    t = time.time();
    print('load data for oribit', orbit_id, flush=True)

    # we add the info about walnut and orbit ID
    data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id), 'Projections', 'tubeV{}'.format(orbit_id))

    # create the numpy array which will receive projection data from tiff files
    projs = np.zeros((len(projs_idx), projs_rows, projs_cols), dtype=np.float32)

    # load the numpy array describing the scan geometry from file
    vecs = np.loadtxt(os.path.join(data_path_full, vecs_name))
    # get the positions we need
    vecs = vecs[projs_idx]

    # load flat-field and dark-fields
    # there are two flat-field images (taken before and after acquisition), we simply average them
    dark = trafo(imageio.imread(os.path.join(data_path_full, dark_name)))
    flat = np.zeros((2, projs_rows, projs_cols), dtype=np.float32)

    for i, fn in enumerate(flat_name):
        flat[i] = trafo(imageio.imread(os.path.join(data_path_full, fn)))
    flat =  np.mean(flat,axis=0)

    # load projection data
    for i in range(len(projs_idx)):
        projs[i] = trafo(imageio.imread(os.path.join(data_path_full, projs_name.format(projs_idx[i]))))

    print(np.round_(time.time() - t, 3), 'sec elapsed', flush=True)

    ### pre-process data ######################################################

    t = time.time();
    print('pre-process data', flush=True)
    # subtract the dark field, divide by the flat field, and take the negative log to linearize the data according to the Beer-Lambert law
    projs -= dark
    projs /= (flat - dark)
    np.log(projs, out=projs)
    np.negative(projs, out=projs)
    # we need to apply some transformations to the projections to get them from
    # the way the scanner reads it out into to way described in the projection
    # geometry and used by ASTRA
    projs = projs[::-1,...]
    projs = np.transpose(projs, (1,0,2))
    projs = np.ascontiguousarray(projs)
    print(np.round_(time.time() - t, 3), 'sec elapsed')

    ### FDL reconstructions on single slices

    # numpy array holding the reconstruction volume
    vol_rec = np.zeros((n_x, n_x, 1), dtype=np.float32)
    # we need to specify the details of the reconstruction space to ASTRA
    # this is done by a "volume geometry" type of structure, in the form of a Python dictionary
    # by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
    vol_geom = astra.create_vol_geom((n_x, 1, n_x))
    vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
    vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
    vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
    vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
    vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
    vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz

    # register both volume and projection geometries and arrays to ASTRA
    vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)

    # construct full path for storing the results
    rad_slice_path_full = os.path.join(rad_slice_path, 'Walnut{}'.format(walnut_id))

    # create the directory in case it doesn't exist yet
    if not os.path.exists(rad_slice_path_full):
        os.makedirs(rad_slice_path_full)


    for i_div in range(n_div):

        t = time.time();
        print('computing FDK reconstruction', i_div+1, '/', n_div, flush=True)

        # rotate astra geometry
        theta_rot = center_angles[i_div]
        vecs_rot  = rotate_astra_vec_geom(vecs, - theta_rot)

        # numpy array holding the reconstruction volume
        vol_rec = np.zeros((n_x, n_x, n_x), dtype=np.float32)

        # we need to specify the details of the reconstruction space to ASTRA
        # this is done by a "volume geometry" type of structure, in the form of a Python dictionary
        # by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
        vol_geom = astra.create_vol_geom((n_x, n_x, n_x))
        vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
        vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
        vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
        vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
        vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
        vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz

        # we need to specify the details of the projection space to ASTRA
        # this is done by a "projection geometry" type of structure, in the form of a Python dictionary
        proj_geom_rot    = astra.create_proj_geom('cone_vec', projs_rows, projs_cols, vecs_rot)

        # register both volume and projection geometries and arrays to ASTRA
        vol_id       = astra.data3d.link('-vol', vol_geom, vol_rec)
        proj_id      = astra.data3d.link('-sino', proj_geom_rot, projs)

        # finally, create an ASTRA configuration.
        # this configuration dictionary setups an algorithm, a projection and a volume
        # geometry and returns a ASTRA algorithm, which can be run on its own
        cfg_fdk = astra.astra_dict('FDK_CUDA')
        cfg_fdk['ProjectionDataId'] = proj_id
        cfg_fdk['ReconstructionDataId'] = vol_id
        cfg_fdk['option'] = {}
        cfg_fdk['option']['ShortScan'] = False
        alg_id = astra.algorithm.create(cfg_fdk)

        # run FDK algorithm
        astra.algorithm.run(alg_id, 1)

        # release memory allocated by ASTRA structures
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)

        print(np.round_(time.time() - t, 3), 'sec elapsed')

        ### extract radial slices

        # set up interpolation
        x_gr         = np.linspace(-1.0, 1.0, n_x)
        interpolator = rgi((x_gr,x_gr,x_gr), vol_rec)

        x_rad        = np.zeros((n_x, n_x))
        y_rad        = np.zeros((n_x, n_x))
        z_rad        = np.zeros((n_x, n_x))

        for i_theta in range(n_rad):

            # check if this slice should be extracted from this volume
            extract_slice =  abs(theta[i_theta] - theta_rot) < angle_div_half_width
            extract_slice =  extract_slice | (abs(theta[i_theta] - (theta_rot + np.pi/2)) < angle_div_half_width)

            if extract_slice:
                for i_z in range(n_x):
                    x_rad[:,i_z]    = x_gr[i_z]
                    y_rad[:,i_z]    = x_gr *  np.cos(theta[i_theta] - theta_rot)
                    z_rad[:,i_z]    = x_gr * -np.sin(theta[i_theta] - theta_rot)

                rad_slice = (interpolator(np.vstack((x_rad.flatten(), y_rad.flatten(), z_rad.flatten())).T)).reshape([n_x,n_x])
                rad_slice = np.float32(rad_slice.T)

                # save slice
                slice_path = os.path.join(rad_slice_path_full, 'fdk_pos{}_ass{}_vmm{}_nut{:02}_{:03}.tiff'.format(
                              orbit_id, angluar_sub_sampling, voxel_per_mm, walnut_id, i_theta))
                imageio.imwrite(slice_path, rad_slice)



print('all FDK reconstructions completed,', np.round_(time.time() - t_fdk, 3), 'sec elapsed')



#### iterative reconstructions of combined data ###############################

### load and pre-process data #################################################

t = time.time();
print('load and pre-process data from all orbits', flush=True)

# we add the info about walnut
data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id), 'Projections')

# Create the numpy array which will receive projection data from tiff files
projs = np.zeros((projs_rows, 0, projs_cols), dtype=np.float32)

# And create the numpy array receiving the motor positions read from the geometry file
vecs = np.zeros((0, 12), dtype=np.float32)

# Loop over the subset of orbits we want to load at the same time
for orbit_id in [1,2,3]:
    orbit_data_path = os.path.join(data_path_full, 'tubeV{}'.format(orbit_id))

    # load the numpy array describing the scan geometry of the orbit from file
    vecs_orbit = np.loadtxt(os.path.join(orbit_data_path, vecs_name))
    # get the positions we need and write into vecs
    vecs = np.concatenate((vecs, vecs_orbit[projs_idx]), axis=0)

    # load flat-field and dark-fields
    # there are two flat-field images (taken before and after acquisition), we simply average them
    dark = trafo(imageio.imread(os.path.join(orbit_data_path, dark_name)))
    flat = np.zeros((2, projs_rows, projs_cols), dtype=np.float32)
    for i, fn in enumerate(flat_name):
        flat[i] = trafo(imageio.imread(os.path.join(orbit_data_path, fn)))
    flat =  np.mean(flat,axis=0)

    # load projection data directly on the big projection array
    projs_orbit = np.zeros((nb_projs_orbit, projs_rows, projs_cols), dtype=np.float32)
    for i in range(len(projs_idx)):
        projs_orbit[i] = trafo(imageio.imread(os.path.join(orbit_data_path, projs_name.format(projs_idx[i]))))

    # subtract the dark field, devide by the flat field, and take the negative log to linearize the data according to the Beer-Lambert law
    projs_orbit -= dark
    projs_orbit /= (flat - dark)

    # take negative log
    np.log(projs_orbit, out=projs_orbit)
    np.negative(projs_orbit, out=projs_orbit)

    # we need to apply some more transformations to the projections to get them from
    # the way the scanner reads it out into to way described in the projection
    # geometry and used by ASTRA
    projs_orbit = projs_orbit[::-1,...]
    projs_orbit = np.transpose(projs_orbit, (1,0,2))

    # attach to projs
    projs = np.concatenate((projs, projs_orbit), axis=1)
    del projs_orbit

projs = np.ascontiguousarray(projs)

print(np.round_(time.time() - t, 3), 'sec elapsed')



### compute iterative reconstructions##########################################

t_iter = time.time();

for i_div in range(n_div):

    t = time.time();
    print('computing iterative reconstruction', i_div+1, '/', n_div, flush=True)

    # rotate astra geometry
    theta_rot = center_angles[i_div]
    vecs_rot  = rotate_astra_vec_geom(vecs, - theta_rot)

    # numpy array holding the reconstruction volume
    vol_rec = np.zeros((n_x, n_x, n_x), dtype=np.float32)

    # we need to specify the details of the reconstruction space to ASTRA
    # this is done by a "volume geometry" type of structure, in the form of a Python dictionary
    # by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
    vol_geom = astra.create_vol_geom((n_x, n_x, n_x))
    vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
    vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
    vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
    vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
    vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
    vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz

    # we need to specify the details of the projection space to ASTRA
    # this is done by a "projection geometry" type of structure, in the form of a Python dictionary
    proj_geom_rot    = astra.create_proj_geom('cone_vec', projs_rows, projs_cols, vecs_rot)

    # register both volume and projection geometries and arrays to ASTRA
    vol_id       = astra.data3d.link('-vol', vol_geom, vol_rec)
    proj_id      = astra.data3d.link('-sino', proj_geom_rot, projs)
    projector_id = astra.create_projector('cuda3d', proj_geom_rot, vol_geom)

    ## finally, create an ASTRA configuration.
    ## this configuration dictionary setups an algorithm, a projection and a volume
    ## geometry and returns a ASTRA algorithm, which can be run on its own
    astra.plugin.register(NesterovGradient.AcceleratedGradientPlugin)
    cfg_agd                            = astra.astra_dict('AGD-PLUGIN')
    cfg_agd['ProjectionDataId']        = proj_id
    cfg_agd['ReconstructionDataId']    = vol_id
    cfg_agd['ProjectorId']             = projector_id
    cfg_agd['option']                  = {}
    cfg_agd['option']['MinConstraint'] = 0
    alg_id                             = astra.algorithm.create(cfg_agd)

    # Run Nesterov Accelerated Gradient Descent algorithm with 'nb_iter' iterations
    nb_iter = 50
    astra.algorithm.run(alg_id, nb_iter)

    # release memory allocated by ASTRA structures
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    print(np.round_(time.time() - t, 3), 'sec elapsed')

    ### extract radial slices

    # set up interpolation
    x_gr         = np.linspace(-1.0, 1.0, n_x)
    interpolator = rgi((x_gr,x_gr,x_gr), vol_rec)


    x_rad        = np.zeros((n_x, n_x))
    y_rad        = np.zeros((n_x, n_x))
    z_rad        = np.zeros((n_x, n_x))

    for i_theta in range(n_rad):

        # check if this slice should be extracted from this volume
        extract_slice =  abs(theta[i_theta] - theta_rot) < angle_div_half_width
        extract_slice =  extract_slice | (abs(theta[i_theta] - (theta_rot + np.pi/2)) < angle_div_half_width)

        if extract_slice:
            for i_z in range(n_x):
                x_rad[:,i_z]    = x_gr[i_z]
                y_rad[:,i_z]    = x_gr *  np.cos(theta[i_theta] - theta_rot)
                z_rad[:,i_z]    = x_gr * -np.sin(theta[i_theta] - theta_rot)

            rad_slice = (interpolator(np.vstack((x_rad.flatten(), y_rad.flatten(), z_rad.flatten())).T)).reshape([n_x,n_x])
            rad_slice = np.float32(rad_slice.T)

            # save slice
            slice_path = os.path.join(rad_slice_path_full, 'iterative_iter{}_ass{}_vmm{}_nut{:02}_{:03}.tiff'.format(
                                      nb_iter, angluar_sub_sampling, voxel_per_mm, walnut_id, i_theta))
            imageio.imwrite(slice_path, rad_slice)


print('all iterative reconstructions completed,', np.round_(time.time() - t_iter, 3), 'sec elapsed')
