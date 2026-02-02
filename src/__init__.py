__version__ = '0.0.4'

from .tensor import Tensor
from .function_loss.MSE import MSE_loss
from .dataloader.dataload import getMnistImage,getMnistLabel
from .layers.layer import Layer
from .layers.dense import Dense
from .logger.build_graph import view_graph
from .optimizer.SGD import SGD

if __name__ == "__main__":
    print(__version__)