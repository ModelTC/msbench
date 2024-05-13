import msbench
from msbench.utils.logger import logger


def enable_sparsification(model):
    logger.info('Enable sparsification.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, msbench.fake_sparse.sparse_base.FakeSparseBase):  # noqa: E501
            logger.debug('Enable sparse: {}'.format(name))
            submodule.enable_fake_sparse()


def disable_sparsification(model):
    logger.info('Disable sparsification.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, msbench.fake_sparse.sparse_base.FakeSparseBase):  # noqa: E501
            logger.debug('Disable sparse: {}'.format(name))
            submodule.disable_fake_sparse()
