import numpy as np
import random
# from tensorflow_addons.image import sparse_image_warp
# import tensorflow_addons #0.20.0
from sparse_image_warp import sparse_image_warp

import nhi_config

# https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html
# Ref: https://arxiv.org/pdf/1904.08779.pdf 

def apply_specaug(features):
    
    '''Apply SpecAugment to features.'''
    seq_len, n_mfcc = features.shape
    outputs = features
    mean_feature = np.mean(features)
    if random.random() < nhi_config.SPECAUG_IMAGE_WARP_PROB: #This time warping augmentation is temporarily disabled atm, prob equals 0 ~(^_^)~
        temp_features = np.reshape(features, (-1, features.shape[0], features.shape[1], 1))
        v, tau = temp_features.shape[1], temp_features.shape[2]
        horizontal_line_through_center = temp_features[0][v//2]
        random_point = horizontal_line_through_center[random.randrange(nhi_config.SPECAUG_IMAGE_WARP_MAX_WIDTH, tau - nhi_config.SPECAUG_IMAGE_WARP_MAX_WIDTH)] # random point along the horizontal/time axis
        w = np.random.uniform((-nhi_config.SPECAUG_IMAGE_WARP_MAX_WIDTH), nhi_config.SPECAUG_IMAGE_WARP_MAX_WIDTH) # distance
        # Source Points
        src_points = [[[v//2, random_point[0]]]]
        
        # Destination Points
        dest_points = [[[v//2, random_point[0] + w]]]
        temp_features, _ = sparse_image_warp(temp_features, src_points, dest_points, num_boundary_points=2)
        outputs = temp_features

    # Frequancy masking.
    if random.random() < nhi_config.SPECAUG_FREQ_MASK_PROB: 
        #---------random.random() return value in [0,1] 
        #------------prob of implement freq/time masking = SPECAUG_FREQ_MASK_PROB = 0.3
        
        width = random.randint(1, nhi_config.SPECAUG_FREQ_MASK_MAX_WIDTH)
        start = random.randint(0, n_mfcc - width)
        outputs[:, start: start + width] = mean_feature

    # Time masking.
    if random.random() < nhi_config.SPECAUG_TIME_MASK_PROB:
        width = random.randint(1, nhi_config.SPECAUG_TIME_MASK_MAX_WIDTH)
        start = random.randint(0, seq_len - width)
        outputs[start: start + width, :] = mean_feature

    return outputs
