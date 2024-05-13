Naive Sparsification-aware Training(SAT)
================================================

The training only requires some additional operations compared to ordinary training process.

.. code-block:: python
    :linenos:

    import torchvision.models as models
    from msbench.scheduler import build_sparse_scheduler

    # first, initialize the FP32 model with pretrained parameters.
    model = models.__dict__["resnet18"](pretrained=True)
    model.train()
    
    # second, build the sparsification training scheduler according to the sparsity_config
    sparse_scheduler = build_sparse_scheduler(sparsity_config)

    # then, we will trace the original model using torch.fx and \
    # replace the origin modules with msbnn.modules
    model = sparse_scheduler.prepare_sparse_model(model=model)

    # here, define the lr_scheduler and optimizer
    lr_scheduler = CosineAnnealingLR(...)
    optimizer = SGD(...)

    if sparsity_config["sparsity"]["scheduler"]["type"] == "AmbaLevelPruneScheduler":
        sparse_scheduler.set_lr_scheduler(lr_scheduler)
    elif sparsity_config["sparsity"]["scheduler"]["type"] == "AmpereScheduler":
        sparse_scheduler.init_optimizer(optimizer)

    model.train()
    # training loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...
        output  = model(input)
        ...
        # do backward procedures
        sparse_scheduler.prune_model(model, iter_idx)
        ...

    # apply mask to layer's weight, and **disable** the fake sparsification process
    sparse_scheduler.export_sparse_model(model)

    # define dummy data for model export.
    dummy_input = torch.randn(1, 3, 224, 224)

    # finally, export the onnx file, ready for deploying to real-world hardware
    torch.onnx.export(model.eval(), dummy_input, "model.onnx", verbose=True, opset_version=10, enable_onnx_checker=True)

..

Now you know how to conduct naive SAT with MSBench, if you want to know more about SparseScheduler check :doc:`../internal/learn_config`.