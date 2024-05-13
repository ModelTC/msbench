import os
import sys
import pdb
from subprocess import Popen, PIPE

lib_path = os.path.dirname(os.path.realpath(__file__))
bin_fn = 'core_sparse_ratio_decision'
bin_path = '{}/{}'.format(lib_path, bin_fn)
if os.path.exists(bin_path) is False:
    print('Error: {} is not in {}'.format(bin_fn, lib_path))
    sys.exit(1)


def get_energy_to_be_pruned(specs_dict=dict()):
    def _sanity_check(input_dict=dict()):
        specs_keys = ['prun_algo', 'prun_algo_tuning', 'given_energy', 'stride_group', 'max_stride_group', 'ctrl_bit', 'is_dw']
        missing_keys = []
        for key in specs_keys:
            if key not in input_dict.keys():
                missing_keys.extend([key])
        if len(missing_keys) > 0:
            print('Error: The input dictionary lack of the following keys {}'.format(missing_keys))
            sys.exit(1)

    def sys_call(cmd_str):
        return Popen(cmd_str, shell=True, stdout=PIPE).communicate()[0]

    _sanity_check(specs_dict)
    argc_list = '{} {} {} {} {} {} {}'.format(specs_dict['prun_algo'], specs_dict['prun_algo_tuning'],
                                              specs_dict['given_energy'], specs_dict['stride_group'], specs_dict['max_stride_group'],
                                              specs_dict['ctrl_bit'], specs_dict['is_dw'])
    cmd_str = '{} {}'.format(bin_path, argc_list)
    try:
        energy = float(sys_call(cmd_str))
        return energy
    except:
        print('AMBA eval version: Fail due to error code: 0xAMB001. Please contact Ambarella')
        sys.exit(1)
