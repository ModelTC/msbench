Learn MSBench configuration
===========================

MSBench provides a primary API **prepare_sparse_model** in sparsification training scheduler class for users to prune their model. 
MSBench will contain various backends presets for **hardware alignment**, but you maybe want to customize your backend.
We provide a guide for learning MSBench configuration, and it will be helpful.

**1.** API **prepare_sparse_model** accepts an extra param, you can provide it following this format.

.. code-block:: python

    sparsity_config = {
        'mask_generator': {
            'type': 'NormalMaskGenerator'                          # custom mask generator
        },
        'fake_sparse': {
            'type': 'FakeSparse'                                   # custom weight fake sparse function
        },
        'scheduler': {
            'type': 'AmbaLevelPruneScheduler'                      # custom sparsification training scheduler
            'kwargs': {
                'total_iters': max_iter,                           # the total number of iterations for sparsification retraining
                'sparsity_table': [30,40,50,60,70,80,90],          # all sparsifying steps, the sparsity should be within [0~100)
                'no_prune_keyword': '',                            # No sparsifying for layer name which has keyword in no_prun_keyword list
                'no_prune_layer': '',                              # No sparsifying for layer name which is exactly listed in the no_prun_layer
                'prun_algo': 1,                                    # For sparsifying algorithm development. Default setting is 1
                'do_sparse_analysis': False,                       # Boolean flag for network analysis (True/False)
                'default_input_shape': [1,3,224,224],              # Used for MAC statistics
                'output_dir': 'path_to_sparse_analysis',           # The output directory for the network analysis and reporter
                'save_dir': 'path_to_sparse_ckpts'                 # The output directory for saving the pruned model ckpts
            }
        }
    }



**2.** **Customize just by:**

.. code-block:: python

    sparse_scheduler = build_sparse_scheduler(sparsity_config)
    model = sparse_scheduler.prepare_sparse_model(model)

**3. Now MSBench support this Mask Generator and Sparser**

**3.1 Mask Generator**


    .. code-block:: markdown
        :linenos:

        NormalMaskGenerator      # More general choice
        ProbMaskGenerator        # Designed for ProbMask Scheduler
        STRMaskGenerator         # Designed for STR Scheduler



**3.2 Sparser**


    .. code-block:: markdown
        :linenos:

        FakeSparse               # Simulate the magnitude-based masking in training time.
        ProbMaskFakeSparse       # Simulate the probabilistic masking during the training time.
        STRFakeSparse            # Simulate the Soft Threshold Weight Reparameterization during the training time.




**4. Recommended sparse training configuration and extra attention**

**4.1 Amba Sparsification-aware Training**

.. code-block:: python

    sparsity_config = {
        'mask_generator': {
            'type': 'NormalMaskGenerator'                          # custom mask generator
        },
        'fake_sparse': {
            'type': 'FakeSparse'                                   # custom weight fake sparse function
        },
        'scheduler': {
            'type': 'AmbaLevelPruneScheduler'                      # custom sparsification training scheduler
            'kwargs': {
                'total_iters': max_iter,                           # the total number of iterations for sparsification retraining
                'sparsity_table': [30,40,50,60,70,80,90],          # all sparsifying steps, the sparsity should be within [0~100)
                'no_prune_keyword': '',                            # No sparsifying for layer name which has keyword in no_prun_keyword list
                'no_prune_layer': '',                              # No sparsifying for layer name which is exactly listed in the no_prun_layer
                'prun_algo': 1,                                    # For sparsifying algorithm development. Default setting is 1
                'do_sparse_analysis': False,                       # Boolean flag for network analysis (True/False)
                'default_input_shape': [1,3,224,224],              # Used for MAC statistics
                'output_dir': 'path_to_sparse_analysis',           # The output directory for the network analysis and reporter
                'save_dir': 'path_to_sparse_ckpts'                 # The output directory for saving the pruned model ckpts
            }
        }
    }

    lr_scheduler = CosineAnnealingLR(...)
    sparse_scheduler = build_sparse_scheduler(sparsity_config)     #  1. define the sparsification training scheduler
    model = sparse_scheduler.prepare_sparse_model(model)           #! 2. trace model and add sparse nodes for model on Unstructured Sparsity Backend
    # Note: when you use AmbaLevelPruneScheduler, you should set the lr_scheduler for sparse_scheduler after you prepare the model
    sparse_scheduler.set_lr_scheduler(lr_scheduler)                #! 3. set lr_scheduler for sparse_scheduler





**4.2 Ampere Sparsification-aware Training**

.. code-block:: python

    sparsity_config = {
        'scheduler': {
            'type': 'AmpereScheduler'                       # ampere sparsification training scheduler
            'kwargs': {
                'mask_calculator': 'm4n2_1d',               # used for generate mask
                'allow_recompute_mask': False,              # Now, we recommend you to choose False
                'allow_permutation': False,                 # Now, we recommend you to choose False
                'verbosity': 0                              # Control the output of intermediate logs
            }
        }
    }
    
    sparse_scheduler = build_sparse_scheduler(sparsity_config)     #  1. define the sparsification training scheduler
    model = sparse_scheduler.prepare_sparse_model(model)           #! 2. trace model and add sparse nodes for model on Unstructured Sparsity Backend
    # Note: when you use AmpereScheduler, you should Initialize the optimizer for sparse_scheduler after you prepare the model
    sparse_scheduler.init_optimizer(optimizer)                     #! 3. Initialize optimizer for sparse_scheduler

    



**4.3 STR Sparsification-aware Training**

.. code-block:: python

    sparsity_config = {
        'mask_generator': {
            'type': 'STRMaskGenerator'                             # custom mask generator
        },
        'fake_sparse': {
            'type': 'STRFakeSparse',                               # custom weight fake sparse function
            'kwargs': {
                'sInit_value': -5                                  # used to control the inintial sparsity
            }
        },
        'scheduler': {
            'type': 'BaseScheduler'                                # custom sparsification training scheduler
        }
    }
