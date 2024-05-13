Advanced PTS
============
This part, we introduce a advanced post-training sparsification methods.
Fair experimental comparisons can be found in Benchmark.

**Post-training Sparsification**

Gradient-based ﬁne-tuning with auxiliary loss functions was previously successfully applied for post-training quantization, 
including AdaRound, BRECQ and QDrop. 

We follow a similar approach whereby we deﬁne a local knowledge distillation loss for every pruned layer[1],
These losses serve as a measure of how close the output activation feature maps of the original (unpruned) 
and pruned layers are. Suppose the pruned layer weights and biases are :math:`W^s` and :math:`b^s` , 
then the knowledge distillation mean-squared error loss for that layer is deﬁned as:


.. raw:: latex html

           \[ L = \sum_{i\in batch}(Y_{dense}^{i} - f(W^{s}M^{s}, X_{dense}^{i}) - b^{s})^{2} \]
            \[s.t. Y_{dense}^{i} = f(W_{dense}, X_{dense}^{i}) \]

where :math:`f(X, M)` is the convolutional/matrix-multiply operation represented by the layer with weights :math:`W`
acting on inputs  :math:`X`. The input tensors :math:`X_{dense}` used to calculate the output activations above are constructed by running a forward pass of the original,
unpruned model. :math:`M^{s}` is the binary mask layer which is equal to one if the corresponding weight is not pruned and zero otherwise.
We ﬁx the sparse binary mask and run a gradient descent of the loss functions deﬁned above for each layer independently to ﬁnd the optimal weights
and biases :math:`M^{s},b^{s}`.



.. code-block:: python
    :linenos:

    import argparse
    from faulthandler import disable
    from data.imagenet import load_data
    from msbench.utils.state import disable_sparsification, enable_sparsification
    from utils import parse_config, seed_all, evaluate
    from msbench.scheduler import build_sparse_scheduler
    from msbench.advanced_pts import pts_reconstruction
    from msbench.utils.logger import logger
    from models import load_model
    import logging
    import time

    # first, initialize the FP32 model with pretrained parameters.
    model = models.__dict__["resnet18"](pretrained=True)

    ## second, build the sparsification training scheduler according to the sparsity_config
    sparse_scheduler = build_sparse_scheduler(sparsity_config)

    # then, we will trace the original model using torch.fx and \
    # replace the origin modules with msbnn.modules
    model = sparse_scheduler.prepare_sparse_model(model=model)

    # disable fake sparse process, and do evaluation for dense model
    disable_sparsification(model)
    evaluate(val_loader, model) # resnet18 * Acc@1 69.758 Acc@5 89.084

    # enable fake sparse process, and do evaluation for sparse model without reconstruction
    enable_sparsification(model)
    evaluate(val_loader, model)  # resnet18 * Acc@1 58.218 Acc@5 81.314

    # set pts config
    pts_config_dict = {
        pattern: layer
        max_count: 20000
        keep_gpu: True
        weight_lr: 0.00001
    }

    logger.info('begin advanced PTS now!')
    start_time = time.time()
    model.train()
    model = pts_reconstruction(model, cali_data, pts_config_dict)
    finish_time = time.time()
    logger.info("pts_reconstruction use time {} mins".format((finish_time-start_time)/60))
    
    # do evaluation
    enable_sparsification(model)
    evaluate(val_loader, model) #  * resnet18 Acc@1 68.968 Acc@5 88.780

    # apply mask to layer's weight, and disable the fake sparsification process
    sparse_scheduler.export_sparse_model(model)

    # define dummy data for model export.
    dummy_input = torch.randn(1, 3, 224, 224)

    # finally, export the onnx file, ready for deploying to real-world hardware
    torch.onnx.export(model.eval(), dummy_input, "model.onnx", verbose=True, opset_version=11, enable_onnx_checker=True)


For the detailed post-training sparsification reconstruction process, 
please refer to the `applications folder <https://gitlab.bj.sensetime.com/spring2/sparsity/-/blob/master/applications/imagenet_example/PTS/pts.py>`_ provided by MSBench



Reference
^^^^^^^^^^^^^^^^^^

`[1]. Post-training deep neural network pruning via layer-wise calibration <https://openaccess.thecvf.com/content/ICCV2021W/LPCV/papers/Lazarevich_Post-Training_Deep_Neural_Network_Pruning_via_Layer-Wise_Calibration_ICCVW_2021_paper.pdf>`_
