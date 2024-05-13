old config file:

sparse:
    USE_ERK: True
    target_sparsity: 0.5
    cali_batchsize: 32
    reconstruction:
        pattern: block
        max_count: 20000
        keep_gpu: True
        weight_lr: 0.00001
    mask_generator:
      type: NormalMaskGenerator
    fake_sparse:
      type: FakeSparse
    scheduler:
      type: BaseScheduler



 new config file:

 sparse:
    cali_batchsize: 32
    reconstruction:
        pattern: block
        max_count: 20000
        keep_gpu: True
        weight_lr: 0.00001
    mask_generator:
      type: NormalMaskGenerator
    fake_sparse:
      type: FakeSparse
    scheduler:
      type: BaseScheduler
      kwargs:
        USE_ERK: True
        target_sparsity: 0.6

 注：两种写法都可以，第二种是应为在BaseScheduler目前已添加了ERK的使用