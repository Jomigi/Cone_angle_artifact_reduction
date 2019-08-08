from msd_pytorch import (MSDRegressionModel)
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from os import environ
from sacred import Experiment, Ingredient
from sacred.observers import MongoObserver
from image_dataset import ImageDataset
import glob
import os.path
import numpy as np
import torch as t
import random
import imageio as io


msd_ingredient = Ingredient("msd")
ex = Experiment('WALNUTS - Radial slices - Validation Experiment', ingredients=[msd_ingredient])

mongo_user = environ.get('MONGO_SACRED_USER')
mongo_pass = environ.get('MONGO_SACRED_PASS')
mongo_host = environ.get('MONGO_SACRED_HOST')

assert mongo_user, 'Setting $MONGO_USER is required'
assert mongo_pass, 'Setting $MONGO_PASS is required'
assert mongo_host, 'Setting $MONGO_HOST is required'

mongo_url = 'mongodb://{0}:{1}@{2}:27017/sacred?authMechanism=SCRAM-SHA-1'.format(
    mongo_user, mongo_pass, mongo_host)

ex.observers.append(MongoObserver.create(url=mongo_url, db_name='sacred'))

#%%
@ex.config
def config():
    
    # Set MSD parameters
    in_channels = 1                     # number of input channels
    out_channels = 1                    # Number of output channels
    depth = 10                          # Depth of the MSD network
    width = 1                           # Width of the MSD network
    reflect = True                      # Use reflection padding
    epochs = 100                        # The number of epochs to train for
    batch_size = 1                      # The mini-batch size
                                        # (1 is strongly recommended for msd_pytorch)
    dilations = [1,2,3,4,5,6,7,8,9,10]  # Dilations of the convolutional kernels in the MSDnet
    position = 2                        # Position of the walnuts when scanned.
                                        # Can be 1,2 or 3. 
    early_stop = 10                     # Number of epochs to wait for improvement of the validation error. 
    train = True                        # Whether the model should be trained
    load_model = False                  # Whether a pre-trained model should be
                                        # loaded
        
    # Training dataset
    dataset_dir = "/bigstore/felix/WalnutsRadialSlices/"          # Where the data is stored
    results_dir = "/export/scratch2/jordi/Data/Walnuts/Results/"  # Where the results are stored
    save_network_path = os.path.join(results_dir, "saved_nets/")  # The the network parameters are stored
    
    selection_seed = 123456             # Select a seed for reproducibility
    input_phantoms = list(range(1,43))  # The walnuts used as input for the MS-D network
    
    # Separate set into training, validation and test set
    training_nb = 28
    val_nb = 7
    test_nb = 0
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
        
    seed = 123654
    
 #%%   
@ex.automain
def main(msd, epochs, in_channels, out_channels, depth, width, reflect, 
         batch_size, dilations, position, early_stop, dataset_dir, save_network_path, 
         results_dir, training_phantoms, val_phantoms, test_phantoms, seed, train, 
         load_model):
    
    np.random.seed(seed)
    
    # Set training set
    inp_imgs = []
    tgt_imgs = []
    for i in sorted(training_phantoms):
        inp_imgs.extend(glob.glob(os.path.join(dataset_dir, 'Walnut{}/*pos{}_*.tif*'
                                                     .format(i, position))))
        tgt_imgs.extend(glob.glob(os.path.join(dataset_dir, 'Walnut{}/iterative_*.tif*'
                                                         .format(i))))
        
    train_ds = ImageDataset(inp_imgs, tgt_imgs)
    print('Training set size', str(len(train_ds)))
    
    # Set validation set
    inp_imgs = []
    tgt_imgs = []
    segm_ds = []

    for i in sorted(val_phantoms):
        inp_imgs.extend(glob.glob(os.path.join(dataset_dir, 'Walnut{}/*pos{}_*.tif*'
                                                         .format(i, position))))
        tgt_imgs.extend(glob.glob(os.path.join(dataset_dir, 'Walnut{}/iterative_*.tif*'
                                                         .format(i))))
          
    val_ds = ImageDataset(inp_imgs, tgt_imgs)
    print('Validation set size', str(len(val_ds)))
    
    # Set test set       
    inp_imgs = []
    tgt_imgs = []
    test_ds = []
  
    for i in sorted(test_phantoms):
        inp_dir = glob.glob(os.path.join(dataset_dir, 'Walnut{}/*pos{}_*.tif*'
                                                     .format(i, position)))
        inp_imgs.extend(inp_dir)
        tgt_dir = glob.glob(os.path.join(dataset_dir, 'Walnut{}/iterative_*.tif*'
                                                     .format(i)))
        tgt_imgs.extend(tgt_dir)
        
        # Create an additional list in order to segment each slice in a later
        # stage. This list is necessary to remember whic slices correspond to 
        # certain walnuts. 
        test_ds.append(ImageDataset(inp_dir, tgt_dir))
                     
    print('Test set size', str(len(ImageDataset(inp_imgs, tgt_imgs))))
        
    # Create dataloaders, which batch and shuffle the data:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    
    # Create MS-D network:
    model = MSDRegressionModel(in_channels, out_channels, depth, width,
                               dilations = dilations, loss = "L2")
    
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
            model.load(saved_network_path + '/Radial_depth{}_epoch{}.pytorch').format(depth, epoch)
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

                       
            end = timer()
            ex.log_scalar("Iteration time", end - startd)
         
            # Save network if worthwile
            if validation_error < best_validation_error:
                best_validation_error = validation_error
                model.save(save_network_path + '/Radial_depth{}_epoch{}.pytorch'.format(
                           depth, epoch), epoch)
                best_epoch = epoch
                early_stop = 0
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
        model.save(save_network_path + 'Radial_depth{}_epoch{}.pytorch'.format(depth, epoch), epoch)
    
    
    #%% Apply the network to the validation set
   
    cor_img = np.zeros((709, 501, 501), dtype=np.float32) # TODO: This shouldn't be hardcoded
    print(len(val_dl))
    
    for walnut_idx, ds in enumerate(test_ds):    
    
        test_dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        for idx, inp in enumerate(test_dl):
            
            #print(inp[0],inp[1])
            model.forward(inp[0],inp[1])
            output = model.output.data.cpu().numpy()
            cor_img[idx, :, :] = output[0,:,:,:]
            
            #Export the result
            path_results = os.path.join(results_dir, 'Radial/pos{}/depth{}/phantom{}/'.format(position, depth, sorted(test_phantoms)[walnut_idx]))
            if not os.path.exists(path_results):
                os.makedirs(path_results)
            io.imsave(path_results + '/slice{:05}.tif'.format(idx), cor_img[idx,:,:].astype("float32"))
