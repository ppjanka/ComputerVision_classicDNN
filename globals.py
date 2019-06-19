# [globals.py] Global variables

# preprocessing-related globals
min_variance = 0.1**2 # a lower cutoff to stddev to filter 'not found' images
min_unique = 100 # minimum number of unique color values for the image

# preprocessed image properties
X_shape = [256,256] # shape of a processed image

# processing properties
batch_size = 64

# allow importing from NNs folder
import sys
sys.path.insert(0, './NNs')