#!/usr/bin/env python3
#
# Copyright (c) 2017-2019 Ambarella, Inc.
#
# This file and its contents ("Software") are protected by intellectual property rights including,
# without limitation, U.S. and/or foreign copyrights.  This Software is also the confidential and
# proprietary information of Ambarella, Inc. and its licensors.  You may not use, reproduce, disclose,
# distribute, modify, or otherwise prepare derivative works of this Software or any portion thereof
# except pursuant to a signed license agreement or nondisclosure agreement with Ambarella, Inc. or
# its authorized affiliates.  In the absence of such an agreement, you agree to promptly notify and
# return this Software to Ambarella, Inc.
#
# THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL AMBARELLA, INC. OR ITS AFFILIATES BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; COMPUTER FAILURE OR MALFUNCTION; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from os import path
from sys import exit
import struct

FN_CRITERION_VER=path.dirname(path.realpath(__file__)) + '/core_sparse_ratio_decision.ver'
FN_AMBACAFFE_VER=path.dirname(path.realpath(__file__)) + '/../../ambacaffe/caffe/ambacaffe.ver'

def get_criterion_verion():
    if not path.isfile(FN_CRITERION_VER):
        print('core_sparse_ratio_decision.ver not found\n')
        exit(1)

    with open(FN_CRITERION_VER, 'rb') as f:
        ver = f.read(1)
        return ord(ver)

def get_ambacaffe_version():
    if not path.isfile(FN_AMBACAFFE_VER):
        print('ambacaffe/caffe/ambacaffe.ver not found\n')
        exit(1)
    
    with open(FN_AMBACAFFE_VER, 'rb') as f:
        ver = f.read(1)
        return ord(ver)
    
def check_criterion_version(user_ver):
    curr_ver = get_criterion_verion()
    if (curr_ver & user_ver) != user_ver:
        print('[INFO] Version conflict: Pruning tool ver {} v.s. criterion ver {}'.format(hex(user_ver), hex(curr_ver)))
        exit(1)
    else:
        print('[INFO] criterion version: {}'.format(hex(curr_ver)))

def check_ambacaffe_version(user_ver):
    curr_ver = get_ambacaffe_version()
    if (curr_ver & user_ver) != user_ver:
        print('[INFO] Version conflict: Pruning tool ver {} v.s. ambacaffe ver {}'.format(hex(user_ver), hex(curr_ver)))
        exit(1)
    else:
        print('[INFO] ambacaffe version: {}'.format(hex(curr_ver)))
