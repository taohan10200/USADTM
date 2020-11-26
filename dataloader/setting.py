from easydict import EasyDict as edict
from config import cfg
__C__ = edict()

cfg_data = __C__





if  cfg.dataset == 'CIFAR10':
    __C__.MEAN_STD=([0.4914, 0.4822, 0.4465],
                       [0.2471, 0.2435, 0.2616])

if  cfg.dataset == 'CIFAR100':
    __C__.MEAN_STD=([0.5071, 0.4867, 0.4408],
                       [0.2675, 0.2565, 0.2761])

if  cfg.dataset == 'MNIST':
    __C__.MEAN_STD = ( [0.1307, 0.1307,0.1307,],
                            [0.3081,0.3081,0.3081])


if  cfg.dataset == 'STL10':
    __C__.MEAN_STD = ( [0.5, 0.5,0.5,],
                            [0.5,0.5,0.5])

if  cfg.dataset == 'SVHN':
    __C__.MEAN_STD = ( [0.5, 0.5,0.5,],
                            [0.5,0.5,0.5])




