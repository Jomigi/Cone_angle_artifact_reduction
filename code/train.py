from msd_pytorch import (MSDRegressionModel)
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from os import environ
from unet_regr_model import UNetRegressionModel
from sacred import Experiment, Ingredient
from sacred.observers import MongoObserver
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
# Author: Jordi Minnema
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
epochs = 60                   
# The mini-batch size
batch_size = 1       
# CNN architecture used('msd' or 'unet')
network='msd'                       

# Dilations of the convolutional kernels in the MS-D Net 
dilations = [1,2,4,8,16]              
# Depth of the MS-D Net
depth = 80            
# Width of the MS-D Net
width = 1                          

# Whether a pre-trained model should be loaded
load_model = False                    
# Whether the model should be trained
train = True                        

##########################################################################################                      
#                           Dataset Parameters                                           #
##########################################################################################

# Path to input and target CBCT scans
dataset_dir = "/bigstore/felix/WalnutsRadialSlices/"   
# Path to store results
results_dir = "/export/scratch2/home/felix/ConeBeamResults/"   
# Path to store network parameters
save_network_path = os.path.join(
        results_dir, "saved_nets/")

# Position of the X-ray source when acquiring the cone-beam CT data (1,2 or 3)
position = 1 
# Iteration number for documentation purposes
it = 1                              
# The CBCT scans used
input_scans = list(range(1,43))   

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
        'Walnut{}/fdk_pos{}_*.tif*'.format(i, position)))))
    tgt_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
        'Walnut{}/iterative_iter50_*.tif*'.format(i)))))

train_ds = ImageDataset(inp_imgs, tgt_imgs)
print('Training set size', str(len(train_ds)))

# Create validation set
inp_imgs = []
tgt_imgs = []

for i in sorted(val_scans):
    inp_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 
        'Walnut{}/fdk_pos{}_*.tif*'.format(i, position)))))
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
        'Walnut{}/fdk_pos{}_*.tif*'.format(i, position))))
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
if network=='msd':
    model = MSDRegressionModel(in_channels, out_channels, depth, width,
                               dilations = dilations, loss = "L2", parallel=True)
# Create U-Net
elif network=='unet':
    model = UNetRegressionModel(in_channels, out_channels, depth, width, 
        loss_function="L2", dilation=dilations, reflect=True, conv3d=False)

#########################################################################################
#                                      Train Model                                      #
#########################################################################################

if train==True:

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
        model.load('path/to/precomputed_network.pytorch')
        print("Network loaded")
    except:
        print("Loading failed")
        pass

# Train network
if train==True:
    print("Training...")
    best_validation_error = 10e6
    start = timer()
    best_epoch = 0
    
    for epoch in range(epochs):
        print("epoch", epoch)
        startd = timer()
        
        # Train        
        model.train(train_dl, 1)
        
        # Compute training error
        train_error = model.validate(train_dl)
        print("    *Training error:", train_error)

        # Compute validation error
        validation_error = model.validate(val_dl)
        print("    *Validation error: {}".format(validation_error))
                   
        endd = timer()
        print('Training time epoch {}: {}'.format(epoch, endd-startd))
     
        # Save network if worthwile
        if validation_error < best_validation_error:
            best_validation_error = validation_error
   
            network_path = save_network_path
            if not os.path.exists(network_path):
                os.makedirs(network_path)
            model.save(network_path + '{}_depth{}_it{}_epoch{}.pytorch'.format(network, depth, it, epoch), epoch)

            best_epoch = epoch
               
    end = timer()
    
    # Print final validation loss and total training time
    print("Final validation loss: {}".format(model.validate(val_dl)))
    print("Total training time:   {}".format(end - start))
    
    # Save network:
    network_path = save_network_path
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    model.save(network_path + '{}_depth{}_it{}_epoch{}.pytorch'.format(network, depth, it, epoch), epoch)

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
                'predictions/{}_pos{}_it{}_depth{}_walnut{}/'.format(
                    network, position, it, depth, sorted(test_scans)[walnut_idx]))
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        io.imsave(path_results + '/slice{:05}.tif'.format(idx), img[idx,:,:].astype("float32"))
    # Print time required for processing
    end = timer()
    print('segmentation time of walnut{}:'.format(test_scans[walnut_idx]), end-start)
