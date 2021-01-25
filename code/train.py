from msd_pytorch import (MSDRegressionModel)
from metrics5 import Metrics
from radial2axial5 import Radial2Axial
from utils import *
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from os import environ
from unet_regr_model_jordi import UNetRegressionModelJordi
from unet_regr_model import UNetRegressionModel
#from sacred import Experiment, Ingredient
#from sacred.observers import MongoObserver
from image_dataset import ImageDataset
from natsort import natsorted
import glob
import os.path
import numpy as np
import torch as t
import random
import imageio as io
import tifffile

# Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam
#-----------------------------------------------------------------
# Author: Jordi Minnema, Shannon Doyle
# Contact: jordi@cwi.nl
# Github: https://github.com/Jomigi/Cone_angle_artifact_reduction
# License: MIT

# This script is intended to train a convolutional neural network 
# (either MS-D Net or U-Net for high cone-angle artifact 
# reduction in walnut CBCT scans.
#----------------------------------------------------------------

##########################################################################################
#                                 Network Parameters                                     #
##########################################################################################

# Number of input channels
in_channels = 1  
# Number of output channels
out_channels = 1                 
# Use of reflection padding
reflect = True                      
# The number of epochs to train
epochs = 10                   
# The mini-batch size
batch_size = 1       
# CNN architecture used('msd' or 'unet')
architecture = 'unet'

# Dilations of the convolutional kernels in the MS-D Net 
dilation_f = False
if dilation_f == 4:
    dilations = [1,2,4,8,16]
#for a densenet
elif dilation_f == 0:
    dilations = [1, 1, 1, 1,1]
# Depth of the Net
depth = 4   #default 80 for MSD-Net, default 4 for U-Net
# Width of the MS-D Net
width = 1                          
loss_f='L2'
lr = 0.001
opt =  'Adam' #or RMSProp
batch_size=1 #default 1 for MSD-Net, default 4 for U-Net

#Parameters for loading pre-trained models
load_model = False            
print(load_model, 'load model')
best_epoch = None #epoch at which model should be loaded
orig_epochs = None #number of epochs the model was trained for originally
orig_it = None  #refers to the label of the run

train=True
metrics_only=False

##########################################################################################                      
#                           Dataset Parameters                                           #
##########################################################################################

# Position of the X-ray source when acquiring the cone-beam CT data (1,2 or 3)
pos = 1
# Iteration number for documentation purposes
it  = None          

# Path to input and target CBCT scans
dataset_dir = "/bigstore/felix/WalnutsRadialSlices/"   
# Path to store results
run_folder= '{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}/'.format(architecture, pos, width, depth, dilation_f,epochs, it)
print(run_folder)
base_path="/bigstore/shannon/"
results_dir = base_path + "ConeBeamResults/"  + run_folder
if not os.path.exists(results_dir):
     os.makedirs(results_dir)  
# Path to store network parameters
network_path = os.path.join(
        base_path, "saved_nets/")

best_models_path= os.path.join(
        network_path, "best_models/")
if not os.path.exists(best_models_path):
	os.makedirs(best_models_path)
run_network_path= network_path + run_folder 
if not os.path.exists(run_network_path):
    os.makedirs(run_network_path)
# The CBCT scans used
input_scans = list(range(1,43))   

if metrics_only ==False:

    ##########################################################################################
    #                     Separate set into training, validation and test set                #
    ##########################################################################################

    # Number of CBCT scans used for training
    training_nb = 28
    # Number of CBCT scans used for validation
    val_nb = 7
    # Number of CBCT scans used for testing
    test_nb = 7

    # Selection seed for reproducibility
    selection_seed = 123456             
    np.random.seed(selection_seed)

    # Determine CBCT scans for training
    training_scans = np.random.choice(
            input_scans, training_nb, replace=False)
    for i in training_scans:
        input_scans.remove(i)
    print(training_scans)
    # Determine CBCT scans for validation
    val_scans = np.random.choice(
            input_scans, val_nb, replace=False)
    for i in val_scans:
        input_scans.remove(i)

    # Determine CBCT scans for testing
    if test_nb > 0:
        test_scans = np.random.choice(input_scans, test_nb, replace=False)
    else:
        test_scans = val_scans

    # Apply random seed
    np.random.seed(selection_seed)

    #########################################################################################
    #                                  Loading Data                                         #
    #########################################################################################

    # Create training set
    inp_imgs = []
    tgt_imgs = []

    for i in sorted(training_scans):
        inp_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
            'Walnut{}/fdk_pos{}_*.tif*'.format(i, pos)))))
        tgt_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
            'Walnut{}/iterative_iter50_*.tif*'.format(i)))))

    train_ds = ImageDataset(inp_imgs, tgt_imgs)
    print('Training set size', str(len(train_ds)))

    # Create validation set
    inp_imgs = []
    tgt_imgs = []

    for i in sorted(val_scans):
        inp_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
            'Walnut{}/fdk_pos{}_*.tif*'.format(i, pos)))))
        tgt_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
            'Walnut{}/iterative_iter50_*.tif*'.format(i)))))
          
    val_ds = ImageDataset(inp_imgs, tgt_imgs)
    print('Validation set size', str(len(val_ds)))

    # Create test set       
    inp_imgs = []
    tgt_imgs = []
    test_ds = []
    test_size = 0

    for i in sorted(test_scans): 
        inp_imgs = natsorted(glob.glob(os.path.join(dataset_dir, 
            'Walnut{}/fdk_pos{}_*.tif*'.format(i, pos))))
        tgt_imgs = natsorted(glob.glob(os.path.join(dataset_dir,
            'Walnut{}/iterative_iter50_*.tif*'.format(i))))
        
        # Create an additional list in order to process 2D slices for evaluation. 
        # This list is necessary to remember which slices correspond to each walnut. 
        test_ds.append(ImageDataset(inp_imgs, tgt_imgs))
        test_size += len(ImageDataset(inp_imgs, tgt_imgs))
                     
    print('Test set size', str(len(ImageDataset(inp_imgs, tgt_imgs))))


    #########################################################################################
    #                                      Create Model                                     #
    #########################################################################################

    # Create MS-D Net 
    if architecture== 'msd':
        model = MSDRegressionModel(in_channels, out_channels, depth, width,
                                   dilations = dilations, loss = loss_f, parallel=True) #find another way to print model
        print(model)
    # Create U-Net
    elif architecture== 'unet':
        model = UNetRegressionModel(run_network_path, in_channels, out_channels, depth, width, 
            loss_function=loss_f, lr=lr, opt=opt, dilation=dilations, reflect=True, conv3d=False)
    elif architecture== 'unet_jordi':
        model = UNetRegressionModelJordi(run_network_path, in_channels, out_channels, depth, width,
        loss_function=loss_f, dilation=dilations, reflect=True, conv3d=False)
    #########################################################################################
    #                                      Train Model                                      #
    #########################################################################################

    if train==True:
        print('Training model..')
        # Define dataloaders
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
     
        # Normalize input data such that it has zero mean and a standard deviation of 1.
        model.set_normalization(train_dl)

        # Print how a random network performs on the validation dataset:
        print("Initial validation loss: {}".format(model.validate(val_dl)))

    # Try loading a precomputed network if wanted:
    if load_model == True:
        try:
            print('trying to load')
            best_model_file ='{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}_epoch{}.pytorch'.format(architecture, pos, width, depth, dilation_f,epochs, it, best_epoch)
            model_folder = '{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}/'.format(architecture, pos, width,depth,dilation_f,orig_epochs,orig_it)
            model_file= '{}_depth{}_it{}_epoch{}.pytorch'.format(architecture, depth,orig_it, best_epoch)
            model.load(network_path +  model_folder + model_file)
            print("Network loaded")
        except:
           print("Loading model failed. Check path.")
           print("Current path: " + network_path +  model_folder + model_file )
           pass

    # Train network
    if train==True:
        print("Training...")
        best_validation_error = 10e6
        start = timer()
        best_epoch = 0
        train_error_list=[]
        val_error_list=[]
        for epoch in range(epochs):
            print("epoch", epoch)
            startd = timer()
            # Train        
            model.train(train_dl, 1)
            
            # Compute training error
            train_error = model.validate(train_dl)
            print("    *Training error:", train_error)
            train_error_list.append(train_error)
            # Compute validation error
            validation_error = model.validate(val_dl)
            print("    *Validation error: {}".format(validation_error))
            val_error_list.append(validation_error)
            endd = timer()
            print('Training time epoch {}: {}'.format(epoch, endd-startd))
         
            # Save network if worthwile
            if validation_error < best_validation_error:
                best_validation_error = validation_error

                best_epoch = epoch
            model.save(run_network_path + '{}_depth{}_it{}_epoch{}.pytorch'.format(architecture, depth, it, epoch), epoch)                   
        end = timer()
        
        # Print final validation loss and total training time
        val_loss=best_validation_error
        train_time= end-start
        print("Total training time: {}".format(train_time))
        plot_loss(epochs, run_network_path, train_error_list, loss_f, 'train')
        plot_loss(epochs, run_network_path, val_error_list, loss_f, 'val')

        print("- best_val_epoch: %f" % best_epoch)
        print("- val_loss: %f" % val_loss)
        print("- train_time: %f" % train_time)
        print("- val_loss: {}".format(val_loss))

        # Save network:
        model.save(best_models_path + '{}_pos{}_width{}_depth{}_dil{}_ep{}_it{}_epoch{}.pytorch'.format(architecture, pos, width, depth, dilation_f,epochs, it, best_epoch), best_epoch)

    #########################################################################################
    #                               Apply trained network to the test set                   #
    #########################################################################################

    # Allocate processed image
    img = np.zeros((709, 501, 501), dtype=np.float32) 

    # Iterate over walnuts scans in test set
    for walnut_idx, ds in enumerate(test_ds):    

        test_dl = DataLoader(ds, batch_size=1, shuffle=False)
        start = timer() 

        # Iterate over each slice
        for idx, inp in enumerate(test_dl):        
            model.forward(inp[0],inp[1])
            output = model.output.data.cpu().numpy()
            img[idx, :, :] = output[0,:,:,:]
            
            # Save the result
            path_results = os.path.join(results_dir, 
                    '{}_pos{}_it{}_depth{}_phantom{}/'.format(
                        architecture, pos, it, depth, sorted(test_scans)[walnut_idx])) #we could only keep the phantom as name
            if not os.path.exists(path_results):
                os.makedirs(path_results)
            io.imsave(path_results + '/slice{:05}.tif'.format(idx), img[idx,:,:].astype("float32"))
        # Print time required for processing
        end = timer()
        print('Processing time of walnut{}:'.format(test_scans[walnut_idx]), end-start)
    #########################################################################################
    #                               Generate axial from radial images                  #
    #########################################################################################

    phantoms = test_scans # [9, 13,16,19,33,36,37]
    for phantom in phantoms:
        radial_phantom_generator=Radial2Axial(phantom, architecture, [depth],  [pos], [it], base_path, run_folder)
        radial_phantom_generator.generate_rad_2_ax()

    #########################################################################################
    #                               Calculate metrics                  #
    #########################################################################################

phantoms = [9, 13,16,19,33,36,37] #test_scans
metrics_loader= Metrics(phantoms, architecture, pos, [it], depth, width, dilation_f, base_path, run_folder) 
SSIM,SSIM_ROI,MSE, MSE_ROI, DSC_low, DSC_ROI_low, DSC_high, DSC_ROI_high, PSNR, PSNR_ROI= metrics_loader.calculate_metrics()
print("- SSIM: %f" % SSIM)
print("- SSIM_ROI: %f" % SSIM_ROI)
print("- MSE: %f" % MSE)
print("- MSE_ROI: %f" % MSE_ROI)
print("- DSC_low: %f" % DSC_low)
print("- DSC_ROI_low: %f" % DSC_ROI_low)
print("- DSC_high: %f" % DSC_high)
print("- DSC_ROI_high: %f" % DSC_ROI_high)
print("- PSNR: %f" % PSNR)
print("- PSNR_ROI: %f" % PSNR_ROI)


