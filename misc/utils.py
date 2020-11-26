import numpy as np
import os
import math
import time
import random
import shutil

import torch
from torch import nn


import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

import pdb


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print (m)

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def logger(exp_path, exp_name, work_dir, exception):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')


    copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)


    return writer, log_file



def logger_for_CMTL(exp_path, exp_name, work_dir, exception):
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    if not os.path.exists(exp_path+ '/' + exp_name):
        os.mkdir(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')


    copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)


    return log_file

def logger_txt(log_file,epoch,scores):

    acc1,acc5,loss = scores

    snapshot_name = 'all_ep_%d_acc1_%.2f_acc5_%.2f' % (epoch + 1, acc1, acc1)

    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('    [acc1 %.2f acc5 %.2f], [val loss %.4f]\n' % (acc1, acc5, loss))
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')    


def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
        if idx>1:# show only one group
            break
        pil_input = restore(tensor[0])
        pil_output = torch.from_numpy(tensor[1]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy()*255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)




def update_model(net,optimizer, scheduler,epoch, i_tb, exp_path,exp_name,scores,train_record,log_file=None):

    acc1 = scores

    snapshot_name = 'all_ep_%d_acc1_%.3f' % (epoch + 1,acc1)

    if acc1 > train_record['best_acc1']:
        train_record['best_acc1'] = acc1
        train_record['last_model_name'] = train_record['best_model_name']
        train_record['best_model_name'] = snapshot_name
        if log_file is not None:
            logger_txt(log_file,epoch,scores)
        last_model_path = os.path.join(exp_path, exp_name, train_record['last_model_name'] + '.pth')
        if os.path.exists(last_model_path):
            os.remove(last_model_path)

        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(exp_path, exp_name, snapshot_name + '.pth'))


    latest_state = {'train_record':train_record, 'net':net.state_dict(), 'optimizer':optimizer.state_dict(),\
                    'scheduler':scheduler.state_dict(), 'epoch': epoch, 'i_tb':i_tb, 'exp_path':exp_path, \
                    'exp_name':exp_name}

    torch.save(latest_state,os.path.join(exp_path, exp_name, 'latest_state.pth'))
    return train_record


def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter.avg) for meter in self.meters]
        print(entries)
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct[:1])
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):        
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, cur_val, class_id):
        self.cur_val[class_id] = cur_val
        self.sum[class_id] += cur_val
        self.count[class_id] += 1
        self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def print_summary(exp_name,scores,train_record):
    acc1= scores
    print ('='*50)
    print (exp_name)
    print ('    '+ '-'*20)
    print ('    [acc1 %.4f ]' % (acc1))
    print ('    '+ '-'*20)
    print ('[best] [model: %s] , [acc1 %.4f]' % (train_record['best_model_name'],train_record['best_acc1']))
    print ('='*50)






