from easydict import EasyDict as edict
import time
__C = edict()
cfg = __C
__C.dataset = 'CIFAR10'   # 'CIFAR10' ,'CIFAR100', 'STL10', 'SVHN'

__C.n_classes = 10
__C.n_labeled = 250     #option: CIFAR10[40, 250,  4000]  CIFAR100[400,2500,10000] STL10, SVHN[40, 250,  1000]
__C.batch_size = 64     # train batch size of labeled samples

__C.gpu_id = '0'
__C.start_add_samples_epoch = 20  # For CIFAR10:20;  For CIFAR100: 50

__C.wresnet_k = 2      # width factor of wide resnet
__C.wresnet_n = 28     # depth of wide resnet
__C.n_epoches = 1024
__C.mu = 7             # factor of train batch size of unlabeled samples
__C.thr = 0.85         # pseudo label threshold
__C.n_imgs_per_epoch = __C.batch_size*1024  # number of training images for each epoch
__C.lam_u = 1.          # coefficient of unlabeled loss
__C.ema_alpha = 0.999   # decay rate for ema module
__C.lr = 0.03           # learning rate for training
__C.weight_decay = 5e-4 #weight decay for optimizer
__C.momentum=0.9        #momentum for optimizer
__C.seed = 3500

__C.resume = False
__C.resume_model = './exp/11-25_22-57_SVHN_250_0.85_0.03/latest_state.pth'  # the path of the resume model


now = time.strftime("%m-%d_%H-%M", time.localtime())
__C.exp_name = now \
			 + '_' + __C.dataset\
			 + '_' + str(__C.n_labeled)\
             + '_' + str(__C.thr)\
             + '_' + str(__C.lr)

__C.exp_path = './exp' # the path of logs, checkpoints, and current codes