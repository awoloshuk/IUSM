################################################################################
#
# This code was taken from 
# v2, 28/11/2018
# https://github.com/choosehappy/PytorchDigitalPathology/blob/master/segmentation_epistroma_unet/train_unet.ipynb
#
################################################################################
import torch
import tables

import os,sys
import glob

import PIL
import numpy as np

import cv2 # Converts colorspaces
import matplotlib.pyplot as plt

# Replace this: from sklearn import cross_validation with from sklearn.model_selection import train_test_split
import sklearn.model_selection as model_selection
#from sklearn.model_selection import train_test_split

import sklearn.feature_extraction.image
import random

#################################
# <> ACK! Hardcoded tissue type
dataname = "epistroma"
#dataname = "nuclei"

# <> ACK! Hardcoded path! Replace this!
os.chdir('./epi')
print('Current working directory: ' + os.getcwd())
#################################

gShowImage = 1

# The size of the tiles to extract and save in the database, must be >= to training size
patch_size = 500 
# The distance to skip between patches, 1 indicated pixel wise extraction, patch_size would result in non-overlapping tiles
stride_size = 250 

# The number of pixels to pad *after* resize to image with by mirroring (edge's of patches tend not to be 
# analyzed well, so padding allows them to appear more centered in the patch)
mirror_pad_size=250 

# what percentage of the dataset should be used as a held out validation/testing set
test_set_size=.1 

resize = 1 #resize input images

# What classes we expect to have in the data
# We have only 2 classes but we could add additional classes and/or specify an index from which we would like to ignore
classes = [0,1] 

#-----Note---
#One should likely make sure that  (nrow+mirror_pad_size) mod patch_size == 0, where nrow is the number of rows after resizing
#so that no pixels are lost (any remainer is ignored)

seed = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the cross validation setup
random.seed(seed) # set the seed
print("random seed (note down for reproducibility): " + str(seed))

# data type in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
img_dtype = tables.UInt8Atom()  

# Create an atom to store the filename of the image, just incase we need it later,
filenameAtom = tables.StringAtom(itemsize=255) 

# Create a list of the mask files.
maskFileList = glob.glob('masks/*.png') 

# Create training and validation stages and split the files appropriately between them
# This just makes 2 list of files, one for training and one for validation.
phases = {}
phases["train"], phases["val"] = next(iter(model_selection.ShuffleSplit(n_splits=1, test_size=test_set_size).split(maskFileList)))

# Specify that we'll be saving 2 different image types to the database, an image and its associated masked
imgtypes = ["img", "mask"]

# This is an associative array that holds
#    storage["filename"] - pathname of the file.
storage = {} 

# block shape specifies what we'll be saving into the pytable array
# Here we assume that masks are 1d and images are 3d
block_shape = {} 
block_shape["img"] = np.array((patch_size, patch_size, 3))
block_shape["mask"] = np.array((patch_size, patch_size)) 

# we can also specify filters, such as compression
filters = tables.Filters(complevel=6, complib='zlib') 


################################################################################
# Process each phase (training, validation) separately.
for phase in phases.keys(): 
    print("Process all files for phase: " + phase)

    # Keep counts of all the classes in for in particular training, since we 
    totals = np.zeros((2, len(classes))) 
    totals[0,:] = classes # can later use this information to create better weights

    # Open the pytable file for this phase (train, validate) and create the runtime storage array
    name_4_hdf5 = dataname+"_"+phase
    hdf5_file = tables.open_file("./"+name_4_hdf5+".pytable", mode='w') 
    storage["filename"] = hdf5_file.create_earray(hdf5_file.root, 'filename', filenameAtom, (0,)) 
    # print("Debug - storage[filename] = " + str(storage["filename"]))

    # Create an associated earray for each of the image types (mask and image).
    # This is an array inside the current pytable file.
    # We keep a reference to this array in the storage array.
    for imgtype in imgtypes: 
        storage[imgtype] = hdf5_file.create_earray(
                                            hdf5_file.root, 
                                            imgtype,
                                            img_dtype,  
                                            shape = np.append([0],block_shape[imgtype]), 
                                            chunkshape = np.append([1],block_shape[imgtype]),
                                            filters = filters)

    # Read each file for the current phase
    for fileIndex in phases[phase]: 
        maskFileName = maskFileList[fileIndex] 
        #print("Process Mask File: " + maskFileName)

        # Process both the original image and the mask for each image/mask pair.
        # We enumerate the files in the masks subdir, and for each mask file we derive the 
        # corresponding image file pathname.
        for imgtype in imgtypes:
            print("Image: " + str(imgtype))

            # The original image is 3 channel.
            # cv2 won't load it in the correct channel order, so we need to fix that
            if (imgtype == "img"): 
                # Derive the corresponding image file pathname.
                imageFileName = os.getcwd() + "/" + os.path.basename(maskFileName).replace("_mask.png", ".tif")
                print("Process image File: " + imageFileName + ", file exists:" + str(os.path.exists(imageFileName)))
                io = cv2.cvtColor(cv2.imread(imageFileName), cv2.COLOR_BGR2RGB)
                interp_method = PIL.Image.BICUBIC                
            # Otherwise, it is a mask image
            else: 
                print("Process Mask File: " + maskFileName)

                # We only need one channel for a mask.
                # The image is loaded as values that range from 0 - 255, but convert it to binary {0,1} since this makes
                # using the mask easier
                io = cv2.imread(maskFileName) / 255 

                # Use nearest! otherwise resizing may cause non-existing classes to be produced via interpolation (e.g., ".25")
                interp_method=PIL.Image.NEAREST 

                # Sum the number of pixels in all files
                # This is done pre-resize, the proportions don't change which is what matters
                for i,key in enumerate(classes): 
                    totals[1,i] += sum(sum(io[:,:,0] == key))
            # End: if (imgtype == "img"): else:

            # Resize the image pixel array.
            io = cv2.resize(io, (0,0), fx = resize, fy = resize, interpolation = interp_method) 
            # Pad the borders. This provides neighboring pixels on all sides when we process the pixels on the
            # edge of the original image.
            io = np.pad(io, [(mirror_pad_size, mirror_pad_size), (mirror_pad_size, mirror_pad_size), (0, 0)], mode="reflect")

            # Convert input image into overlapping tiles, 
            # For images, the tile size is patch_size x patch_size x 3
            io_arr_out = sklearn.feature_extraction.image.extract_patches(io, (patch_size, patch_size, 3),stride_size)
            # Resize each tile into an array that is patch_size x patch_size x 3
            io_arr_out = io_arr_out.reshape(-1, patch_size, patch_size, 3)

            # Save the 4D tensor to the table. We only need 1 channel for mask data
            if (imgtype == "img"):
                storage[imgtype].append(io_arr_out)
            else:
                storage[imgtype].append(io_arr_out[:,:,:,0].squeeze())
        # End: for imgtype in imgtypes:

        # Add the filename to the storage array
        storage["filename"].append([maskFileName for x in range(io_arr_out.shape[0])]) 
    # End: for fileIndex in phases[phase]:

    #lastely, we should store the number of pixels
    npixels = hdf5_file.create_carray(hdf5_file.root, 'numpixels', tables.Atom.from_dtype(totals.dtype), totals.shape)
    npixels[:] = totals

    # Save and close the database file
    hdf5_file.close()
# End: for phase in phases.keys():     

print("Finished successfully!")

