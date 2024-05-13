#!/usr/bin/env python3
import numpy as np
import os
import sys
import argparse
import pdb

def compute_density(code_name, level_agg, is_dw, channels_out, channels_in, kernel_h, kernel_w, kernel_dec_h, kernel_dec_w, pool_dec_h=1, pool_dec_w=1):
    '''
    v1.6
    '''
    is_cv2 = code_name == 'cv2'
    is_cv2a = code_name == 'cv2fs'

    channels = (channels_out/channels_in) if is_dw else channels_out
    d_kernel, d_in, d_out = 8, 8, 8
    if len(level_agg)==3:
        d_kernel, d_in, d_out = level_agg
    d_tr = 8 if d_in==8 and d_out==8 and d_kernel==8 else 16

    factor_0_c = 1 if is_cv2 else 2
    factor_1_c = 0.046875
    factor_2_eq = 0.03125 * (channels_in/channels_out) * kernel_dec_h * (d_in/d_tr)
    factor_3_c = 0.03515625 if is_cv2a else 0.0234375
    factor_3_eq = (channels_in/channels_out) * kernel_dec_h * kernel_dec_w * pool_dec_h * pool_dec_w * (d_in/d_out)
    factor = factor_0_c * min(factor_1_c, factor_2_eq, factor_3_c * factor_3_eq)

    density = 1/(factor * kernel_w * kernel_h * channels * (d_kernel/8))
    return density

def compute_target_density(cv_chip, level_agg, is_dw, channels_out, channels_in, kernel_h, kernel_w, kernel_dec_h, kernel_dec_w, pool_dec_h=1, pool_dec_w=1):
    '''
    HW-FastConv will be idle when the calculation is bounded by IO.
    To make sure HW-FastConv is fully utilized, density of the kernel is better to larger than target_density

    Input: cv_chip (as target_code_mapping.keys() shows), level_agg, is_dw (is depthwise convolution), channels_out/channels_in/kernel_h/kernel_w/kernel_dec_h/kernel_dec_w are related to kernel size

    Return target_density
    '''
    # code name handling
    target_code_mapping = { 'cv2':    'cv2',
                            'cv2aq':  'cv2',

                            'cv22':   'cv22',
                            'cv22aq': 'cv22',
                            'cv25':   'cv22',
                            'cv25aq': 'cv22',

                            'cv2fs':  'cv2fs',
                            'cv22fs': 'cv2fs' }
    if isinstance(cv_chip, str) is not True:
        print('The chip ID should be string format')
        sys.exit(1)
    cv_chip_lower = cv_chip.lower()
    if cv_chip_lower not in target_code_mapping.keys():
        print('Please make sure the chip ID {} is in supporting list {}'.format(cv_chip_lower, target_code_mapping.keys()))
        sys.exit(1)

    code_name = target_code_mapping[cv_chip_lower]
    target_density = 1
    density = compute_density(code_name, level_agg, is_dw, channels_out, channels_in, kernel_h, kernel_w, kernel_dec_h, kernel_dec_w, pool_dec_h, pool_dec_w)
    target_density = min(1, density)
    return target_density

def get_min_split_length(dilated_length, dilation):
    DILATION_SIZE_TRIGGER_SPLIT = 5
    KERNEL_SIZE_LIMIT = 11
    start_p = 0
    min_length = 0
    if dilation > DILATION_SIZE_TRIGGER_SPLIT:
        if (dilation+1) > KERNEL_SIZE_LIMIT:
            min_length = 1
        else:
            while start_p <  dilated_length:
                min_length = min(dilation+1, dilated_length-start_p)
                start_p += min_length + dilation - 1
    else:
        min_length = dilated_length % KERNEL_SIZE_LIMIT
        min_length = min_length if min_length else KERNEL_SIZE_LIMIT
    return min_length

def get_min_split_kernel_size(dilated_kernel_h, dilated_kernel_w, dilation_h, dilation_w=0):
    '''
    To fully use HW-FastConv, CnnGen might split kernel into smaller one

    Input: dilated_kernel_h, dilated_kernel_w, dilation_h, dilation_w

    Return min_h, min_w (minimum split kernel size)
    '''
    min_h = get_min_split_length(dilated_kernel_h, dilation_h)
    min_w = get_min_split_length(dilated_kernel_w, dilation_h if dilation_w==0 else dilation_w)
    return min_h, min_w

def get_dilated_kernel_size(kernel_h, kernel_w, dilation_h, dilation_w=0, fwk=0):
    '''
    Input: kernel_h, kernel_w, dilation_h, dilation_w, fwk (different framework might have different calculation, 0: caffe)

    Return: dilated_kernel_h, dilated_kernel_w
    '''
    dilated_kernel_h = kernel_h + (kernel_h-1)*(dilation_h-1)
    dilated_kernel_w = kernel_w + (kernel_w-1)*((dilation_h if dilation_w==0 else dilation_w)-1)
    return dilated_kernel_h, dilated_kernel_w

def main(argv):
    parser = argparse.ArgumentParser(description="HW target sparsity/density analysis")
    args = parser.parse_args()

if __name__=='__main__':
    main(sys.argv[1:])
