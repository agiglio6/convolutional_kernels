import numpy as np

from .conv_kernel import CConvKernel


class CConvKernelMovingAverage(CConvKernel):

    def __init__(self):
        super().__init__()

    def kernel_mask(self):
        x = np.ones(self.kernel_size)
        self._mask = x / np.sum(x)
