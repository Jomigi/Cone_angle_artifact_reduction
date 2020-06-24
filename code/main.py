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

msd_ingredient = Ingredient("msd")
ex = Experiment('Experiment_Name', ingredients=[msd_ingredient])

mongo_user = environ.get('MONGO_SACRED_USER')
mongo_pass = environ.get('MONGO_SACRED_PASS')
mongo_host = environ.get('MONGO_SACRED_HOST')

assert mongo_user, 'Setting $MONGO_USER is required'
assert mongo_pass, 'Setting $MONGO_PASS is required'
assert mongo_host, 'Setting $MONGO_HOST is required'

mongo_url = 'mongodb://{0}:{1}@{2}:27017/sacred?authMechanism=SCRAM-SHA-1'.format(
mongo_user, mongo_pass, mongo_host)

ex.observers.append(MongoObserver.create(url=mongo_url, db_name='sacred'))

@ex.config
def config():
    
    # Set CNN parameters
    in_channels = 1                     # Number of input channels
    out_channels = 1                    # Number of output channels
    reflect = True                      # Use reflection padding
    epochs = 60                         # The number of epochs to train for (MSDnet: 60 ; U-Net: 10) 
    batch_size = 1                      # The mini-batch size
                                        # (1 is strongly recommended for msd_pytorch)
    network='msd'

    # MS-D network parameters                                        
    dilations = [1,2,4,8,16]            # Dilations of the convolutional kernels in the MSDnet
    depth = 80                          # Depth of the MSD network
    width = 1                           # Width of the MSD network

    # Training parameters
    position = 1                        # Position of the X-ray source trajectory position (1,2 or 3). 
    it = 1                              # Iteration number (for documentation purposes)
    early_stop = 1000                   # Number of epochs to wait for improvement of the validation error. 
    train = True                        # Whether the model should be trained
    load_model = False                  # Whether a pre-trained model should be
                                        # loaded 
    # Paths
    dataset_dir = "/path/to/dataset/"   # Path to input and target data
    results_dir = "/path/to/results/"   # Where to store results
    save_network_path = os.path.join(results_dir, "saved_nets/")  # The the network parameters are stored
    
    selection_seed = 123456             # Select a seed for reproducibility
    input_phantoms = list(range(1,43))  # The walnuts used as input for the MS-D network
    
    # Separate set into training, validation and test set
    training_nb = 28
    val_nb = 7
    test_nb = 7
    np.random.seed(selection_seed)
    training_phantoms = np.random.choice(input_phantoms, training_nb, 
                                         replace=False)
    for i in training_phantoms:
        input_phantoms.remove(i)
    
    val_phantoms = np.random.choice(input_phantoms, val_nb, replace=False)
    for i in val_phantoms:
        input_phantoms.remove(i)
    
    if test_nb > 0:
        test_phantoms = np.random.choice(input_phantoms, test_nb, replace=False)
    else:
        test_phantoms = val_phantoms

@ex.automain
def main(msd, network, epochs, in_channels, out_channels, depth, width, reflect, 
         batch_size, dilations, position, early_stop, dataset_dir, save_network_path, 
         results_dir, training_phantoms, val_phantoms, test_phantoms, seed, train, 
         load_model, it):
    
    np.random.seed(seed)
    
    # Set training set
    inp_imgs = []
    tgt_imgs = []
    for i in sorted(training_phantoms):
            inp_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 'Walnut{}/Reconstructions/fdk_pos{}_*.tif*'
                                                        .format(i, position)))))
            tgt_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 'Walnut{}/Reconstructions/nnls_pos123_*.tif*'
                                                         .format(i)))))

    train_ds = ImageDataset(inp_imgs, tgt_imgs)
    print('Training set size', str(len(train_ds)))
    
    # Set validation set
    inp_imgs = []
    tgt_imgs = []
    segm_ds = []

    for i in sorted(val_phantoms):
        inp_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 'Walnut{}/Reconstructions/fdk_pos{}_*.tif*'
                                                        .format(i, position)))))
        tgt_imgs.extend(natsorted(glob.glob(os.path.join(dataset_dir, 'Walnut{}/Reconstructions/nnls_pos123_*.tif*'
                                                         .format(i)))))
          
    val_ds = ImageDataset(inp_imgs, tgt_imgs)
    print('Validation set size', str(len(val_ds)))
    
    # Set test set       
    inp_imgs = []
    tgt_imgs = []
    test_ds = []
  
    for i in sorted(test_phantoms): 
        inp_dir = natsorted(glob.glob(os.path.join(dataset_dir, 'Walnut{}/Reconstructions/fdk_pos{}_*.tif*'
                                                 .format(i, position))))
        inp_imgs.extend(inp_dir)
        tgt_dir = (glob.glob(os.path.join(dataset_dir, 'Walnut{}/Reconstructions/nnls_pos123_*.tif*'
                                                 .format(i))))
        tgt_imgs.extend(tgt_dir)
        
        # Create an additional list in order to process individual slices for evaluation. 
        # This list is necessary to remember whic slices correspond to certain walnuts. 
        test_ds.append(ImageDataset(inp_dir, tgt_dir))
                     
    print('Test set size', str(len(ImageDataset(inp_imgs, tgt_imgs))))
           
    # Create CNN 
    if network=='msd':
        model = MSDRegressionModel(in_channels, out_channels, depth, width,
                                   dilations = dilations, loss = "L2", parallel=True)
    
    elif network=='unet':
        model = UNetRegressionModel(in_channels, out_channels, depth, width, loss_function="L2", dilation=dilations, reflect=True, conv3d=False)

    if train==True:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
     
        # The network works best if the input data has mean zero and has a
        # standard deviation of 1. To achieve this, we get a rough estimate of
        # correction parameters from the training data. These parameters are
        # not updated after this step and are stored in the network, so that
        # they are not lost when the network is saved to and loaded from disk.
        model.set_normalization(train_dl)
    
        # Print how a random network performs on the validation dataset:
        print("Initial validation loss: {}".format(model.validate(val_dl)))
    
    # Try loading a precomputed network:
    if load_model == True:
        try:
            model.load('path/to/precomputed_network.pytorch.pytorch')
            print("Network loaded for application to final validation set")
        except:
            print("Loading failed")
            pass

    # Train network
    if train==True:
        print("Training...")
        best_validation_error = 10e6
        start = timer()
        best_epoch = 0
        earlystop = 0
        
        for epoch in range(epochs):
            print("epoch", epoch)
            startd = timer()
            
            # Train        
            model.train(train_dl, 1)
            
            # Compute training error
            train_error = model.validate(train_dl)
            print("    *Training error:", train_error)
            ex.log_scalar("Training error", train_error)

            # Compute validation error
            validation_error = model.validate(val_dl)
            print("    *Validation error: {}".format(validation_error))
            ex.log_scalar("Validation error", validation_error)

                       
            endd = timer()
            print('Training time epoch {}: {}'.format(epoch, endd-startd)
         
            # Save network if worthwile
            if validation_error < best_validation_error:
                best_validation_error = validation_error
       
                network_path = save_network_path
                if not os.path.exists(network_path):
                    os.makedirs(network_path)
                model.save(network_path + '{}_depth{}_it{}_epoch{}.pytorch'.format(network, depth, it, epoch), epoch)

                best_epoch = epoch
                earlystop = 0
            else:
                ''' Early stopping. Stop training when the validation error has not improved for 10 epochs. '''
                earlystop += 1
                if earlystop >= early_stop:
                    print("Early-stopping condition has been met.")
                    break
    
        end = timer()
        
        print("Final validation loss: {}".format(model.validate(val_dl)))
        print("Total training time:   {}".format(end - start))
        
        # Save network:
        network_path = save_network_path
        if not os.path.exists(network_path):
            os.makedirs(network_path)
        model.save(network_path + '{}_depth{}_it{}_epoch{}.pytorch'.format(network, depth, it, epoch), epoch)

    #%% Apply the network to the test set
    cor_img = np.zeros((501, 501, 501), dtype=np.float32) # TODO: This shouldn't be hardcoded
    
    
    for walnut_idx, ds in enumerate(test_ds):    
    
        test_dl = DataLoader(ds, batch_size=1, shuffle=False)
        start = timer() 
        for idx, inp in enumerate(test_dl):
            
            #print(inp[0],inp[1])
            model.forward(inp[0],inp[1])
            output = model.output.data.cpu().numpy()
            cor_img[idx, :, :] = output[0,:,:,:]
            
            #Export the result
            path_results = os.path.join(results_dir, 'predictions/Horizontal/{}_pos{}_it{}_depth{}_walnut{}/'.format(network, position, it, depth, sorted(test_phantoms)[walnut_idx]))
            if not os.path.exists(path_results):
                os.makedirs(path_results)
            io.imsave(path_results + '/slice{:05}.tif'.format(idx), cor_img[idx,:,:].astype("float32"))
        end = timer()
        print('segmentation time Walnut{}:'.format(test_phantoms[walnut_idx]), end-start)
