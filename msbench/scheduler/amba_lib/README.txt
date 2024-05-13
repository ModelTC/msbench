* core_sparse_ratio_decision: Sparsification criterion

* core_sparse_ratio_decision.ver: One byte file to store version of sparsification criterion
  - 0x20 means version 2.0

* version_check.py: Make sure if sparsification tool match with sparsification criterion and training framework
  - Usage:
        if os.path.isdir(os.path.dirname(os.path.realpath(__file__)) + '/../lib/'):
            sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib/')
        else:
            print('Please check location of ambatrainingtool/lib')
            exit(1)
        from version_check import check_criterion_version, check_ambacaffe_version
        ################################################################################
        # auto_pruning.py should co-work with corresponding version of ambacaffe
        # and criterion, so here is an sanity check to see if the version of both
        # are expected
        check_criterion_version(0x20)
        check_ambacaffe_version(0x21)
        ################################################################################

* hw_analysis.py: Implementation for HW-aware density calculation
  - Use compute_target_density() with kernel info to get HW-aware density
  - Regarding to kernel size, it can be calculated by
    Step1: get_dilated_kernel_size()
    Step2: get_min_split_kernel_size()
    Step3: Feed min_h and min_w into compute_target_density
  - Supported chip is cv2, cv22, cv25, cv2fs
