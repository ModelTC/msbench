Object Detection Benchmark
==========================

Based on `Msbench <https://gitlab.bj.sensetime.com/spring2/sparsity/-/tree/master>`_ and `UP <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/dev>`_ , we provide an object detection benchmark on COCO dataset.


**SparseScheduler**: AmbaLevelPruneScheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **基础模型**


+-----------------------------------------------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| Network(config)/sparsity                                  | dense |  30%  |  40%  |  50%  |  60%  |  70%  |  80%  |  90%  |
+-----------------------------------------------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| retinanet-FPN(resnet50,bs=2*16,epoch=7*12)                | 37.0  |  37.5 | 37.7  | 37.6  | 37.2  |  36.7 |  33.7 |  32.6 |
+-----------------------------------------------------------+-------+-------+-------+-------+-------+-------+-------+-------+



- **高精度 baseline**

+-----------------------------------------------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| Network(config)/sparsity                                  | dense |  30%  |  40%  |  50%  |  60%  |  70%  |  80%  |  90%  |
+-----------------------------------------------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| retinanet-improve(resnet18,bs=8*16,epoch=7*12)            | 40.7  |  40.7 |  40.7 | 40.5  |  40.3 | 39.5  |  37.1 |  29.5 |
+-----------------------------------------------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| retinanet-improve-cos-iou(resnet18,bs=8*16,epoch=7*12)    | 41.3  |  41.3 |  41.2 |  40.9 |  40.9 | 39.8  |  36.7 |  27.6 |
+-----------------------------------------------------------+-------+-------+-------+-------+-------+-------+-------+-------+




.. note::
  All experiments use the following sparsity_config

  .. code-block:: python

    sparsity:
      mask_generator:
        type: NormalMaskGenerator
      fake_sparse:
        type: FakeSparse
      scheduler:
        type: AmbaLevelPruneScheduler
        kwargs:
          total_iters: None
          sparsity_table: [30,40,50,60,70,80,90]
          no_prune_keyword: ''
          no_prune_layer: ''
          prun_algo: 1
          prun_algo_tuning: 0.5
          dw_no_prune: False
          do_sparse_analysis: False
          default_input_shape: [1,3,1333,1333]  # choose your input shape
          output_dir: path_to_sparse_analysis
          save_dir: path_to_sparse_ckpts
    leaf_module: [Space2Depth, FrozenBatchNorm2d]


**SparseScheduler**: AmpereScheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **基础模型**


+-----------------------------------------------------------+-------+-------+
| Network(config)/sparsity                                  | dense |  50%  |
+-----------------------------------------------------------+-------+-------+
| retinanet-FPN(resnet50,bs=32,epoch=12)                    | 37.0  | 36.8  |
+-----------------------------------------------------------+-------+-------+


- **高精度 baseline**

+-----------------------------------------------------------+-------+-------+
| Network(config)/sparsity                                  | dense |  50%  |
+-----------------------------------------------------------+-------+-------+
| retinanet-improve(resnet18,bs=8*16,epoch=12)              | 40.7  |  40.3 |
+-----------------------------------------------------------+-------+-------+
| retinanet-improve-cos-iou(resnet18,bs=8*16,epoch=12)      | 41.3  |  40.8 |
+-----------------------------------------------------------+-------+-------+
| faster-rcnn-FPN-improve(resnet50,bs=2*16,epoch=20)        | 43.5  |  41.7 |
+-----------------------------------------------------------+-------+-------+


.. note::
  All experiments use the following sparsity_config

  .. code-block:: python

    sparsity:
      scheduler:
        type: AmpereScheduler
        kwargs:
          mask_calculator: m4n2_1d
          allow_recompute_mask: False
          allow_permutation: False
          verbosity: 0