Quick Start
=================================================

This page is for researchers **who want to validate their marvelous model sparsification idea using MSBench**.

MSBench is a benchmark, a framework and a good tool for researchers. Model pruning is a technique to reduce the model size and 
computation by reducing model weight size or itermediate state size. It usually has following path:

1. Pre-training a model -> Pruning the model aware training

2. Pruning the model aware training 

MSBench is designed easy-to-use for researchers, 
we provide step-by-step instructions and detailed comments below to help you finish deploying the **PyTorch ResNet-18** model using the **Unstructured Sparsity** Backend.

Before starting, you should install MSBench first. Now we start the tour.


**1**. **To begin with, let's import MSBench and prepare FP32 model.**

.. code-block:: python

    import torchvision.models as models                           # for example model
    from msbench.scheduler import build_sparse_scheduler          # import the sparsification training scheduler

    model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
    model.eval()

**2**. **Then we learn the extra configration to custom Unstructured Sparsity Backend.**

Using AmbaLevelPruneScheduler to prune the model and generating tha masks by the L1Norm of the layer weights. Usually, 
sparsification training scheduler requires **sparsity_config** as parameters.

.. code-block:: python

    sparsity_config = {
        'mask_generator': {
            'type': 'NormalMaskGenerator'                         # custom mask generator
        },
        'fake_sparse': {
            'type': 'FakeSparse'                                  # custom weight fake sparse function
        },
        'scheduler': {
            'type': 'AmbaLevelPruneScheduler',                     # custom sparsification training scheduler
            'kwargs': {
                'total_iters': max_iter,                           # the total number of iterations for sparsification retraining
                'sparsity_table': [30,40,50,60,70,80,90],          # all sparsifying steps, the sparsity should be within [0~100)
                'no_prune_keyword': '',                            # No sparsifying for layer name which has keyword in no_prun_keyword list
                'no_prune_layer': '',                              # No sparsifying for layer name which is exactly listed in the no_prun_layer
                'prun_algo': 1,                                    # For sparsifying algorithm development. Default setting is 1
                'do_sparse_analysis': False,                       # Boolean flag for network analysis (True/False)
                'default_input_shape': [1,3,224,224],              # Used for MAC statistics
                'output_dir': 'path_to_sparse_analysis',           # The output directory for the network analysis and reporter
                'save_dir': 'path_to_sparse_ckpts',                # The output directory for saving the pruned model ckpts
            }
        }
    }

**3**. **The next step prepares to conduct the experiment.**

.. code-block:: python

    lr_scheduler = CosineAnnealingLR(...)
    sparse_scheduler = build_sparse_scheduler(sparsity_config)    #! 0. define the sparsification training scheduler
    model = sparse_scheduler.prepare_sparse_model(model)          #! 1. trace model and add sparse nodes for model on Unstructured Sparsity Backend
    sparse_scheduler.set_lr_scheduler(lr_scheduler)               #! 2. set lr_scheduler for sparse_scheduler, reset the lr_scheduler and save every sparse model under the certain prune step

    # sparsification training loop
    model.train()
    for iter_idx in range(start_iter, max_iter):
        # do forward procedures
        ...
        output  = model(input)
        ...
        # do backward procedures
        sparse_scheduler.prune_model(model, iter_idx)             #! 3. turn on actually sparsification, ready for simulating Backend inference

    # evaluation loop
    model.eval()
    for i, batch in enumerate(test_data):
        # do forward procedures
        ...
    
    # export the sparse model
    sparse_scheduler.export_sparse_model(model)                  #! 4. apply mask to layer's weight, and **disable** the fake sparsification process

    # define dummy data for model export.
    dummy_input = torch.randn(1, 3, 224, 224)                    #! 5. export the onnx file, ready for deploying to real-world hardware
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, opset_version=10, enable_onnx_checker=True)

**Now you can use exported files to test on real hardware(Ambarella CV2x)  using CVflow inference engine, congratulations!**
