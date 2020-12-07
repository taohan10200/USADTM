import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.MI_losses import Triplet_MI_loss
from model.WideResnet import WideResnet
from dataloader.loading_data import loading_data,update_loading
from label_guessor import LabelGuessor
from misc.lr_scheduler import WarmupCosineLrScheduler
from misc.ema import EMA
from config import cfg
from misc.utils import *
import copy

class Trainer():
    def __init__(self,pwd):
        self.model = WideResnet(cfg.n_classes, k=cfg.wresnet_k, n=cfg.wresnet_n, batchsize= cfg.batch_size)
        self.model= self.model.cuda()

        self.unlabeled_trainloader, self.val_loader = loading_data()
        self.labeled_trainloader = update_loading(epoch = 0)

        wd_params, non_wd_params = [], []
        for param in self.model.parameters():
            if len(param.size()) == 1:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        param_list = [{'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
        self.optimizer = torch.optim.SGD(param_list, lr=cfg.lr, weight_decay=cfg.weight_decay,momentum=cfg.momentum, nesterov=True)

        self.n_iters_per_epoch = cfg.n_imgs_per_epoch // cfg.batch_size

        self.lr_schdlr = WarmupCosineLrScheduler(self.optimizer, max_iter=self.n_iters_per_epoch * cfg.n_epoches, warmup_iter=0)

        self.lb_guessor = LabelGuessor(args=cfg)

        self.train_record = {'best_acc1': 0, 'best_model_name': '','last_model_name': ''}
        self.cross_entropy = nn.CrossEntropyLoss().cuda()
        self.i_tb = 0
        self.epoch = 0
        self.exp_name = cfg.exp_name
        self.exp_path = cfg.exp_path
        if cfg.resume:
            print('Loaded resume weights for WideResnet')

            latest_state = torch.load(cfg.resume_model)
            self.model.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.lr_schdlr.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.ema = EMA(self.model, cfg.ema_alpha)
        self.writer,self.log_txt = logger(cfg.exp_path, cfg.exp_name, pwd, ['exp','dataset','pretrained','pre_trained'])

    def forward(self):

        print('start to train')
        for epoch in range(self.epoch,cfg.n_epoches):
            self.epoch=epoch

            print(('='*50+'epoch: {}'+'='*50).format(self.epoch+1))
            self.train()
            torch.cuda.empty_cache()
            self.evaluate(self.ema)

    def train(self):
        if self.epoch > cfg.start_add_samples_epoch:
            indexs, pre_targets = self.lb_guessor.label_generator.new_data()
            self.labeled_trainloader = update_loading(copy.deepcopy(indexs), copy.deepcopy(pre_targets),self.epoch)
            self.lb_guessor.init_for_add_sample(self.epoch, cfg.start_add_samples_epoch)

        self.model.train()
        Loss,Loss_L, Loss_U, Loss_U_Real, Loss_MI=AverageMeter(),AverageMeter(),AverageMeter(),\
                                                  AverageMeter(),AverageMeter()
        Correct_Num,Valid_Num =AverageMeter(), AverageMeter()

        st = time.time()
        l_set, u_set = iter(self.labeled_trainloader), iter(self.unlabeled_trainloader)

        for it in range(self.n_iters_per_epoch):
            (img, img_l_weak, img_l_strong), lbs_l = next(l_set)

            (img_u, img_u_weak, img_u_strong), lbs_u_real, index_u = next(u_set)


            img_l_weak,img_l_strong,lbs_l = img_l_weak.cuda(), img_l_strong.cuda(),lbs_l.cuda()
            img_u,img_u_weak,img_u_strong = img_u.cuda(),img_u_weak.cuda(),img_u_strong.cuda()

            lbs_u, valid_u = self.lb_guessor(self.model, img_l_weak, img_u_weak, lbs_l,index_u)

            n_u =  img_u_strong.size(0)


            img_cat = torch.cat([img_l_weak,img_u,img_u_weak, img_u_strong], dim=0).detach()

            _, __, pred_l, pred_u = self.model(img_cat)

            pred_u_o, pred_u_w, pred_u_s = pred_u[:n_u], pred_u[n_u:2 * n_u], pred_u[2 * n_u:]

            #=====================cross-entropy loss for labeled data==============
            loss_l =  self.cross_entropy(pred_l, lbs_l)

            # =====================T-MI loss for unlabeled data==============
            if self.epoch>=20:
                T_MI_loss = Triplet_MI_loss(pred_u_o,pred_u_w,pred_u_s)
            else:
                T_MI_loss = torch.tensor(0)

            # =====================cross-entropy loss for unlabeled data==============
            if lbs_u.size(0)>0 and self.epoch>=2:
                pred_u_s = pred_u_s[valid_u]
                loss_u = self.cross_entropy(pred_u_s, lbs_u)

                with torch.no_grad():
                    lbs_u_real = lbs_u_real[valid_u].cuda()
                    valid_num = lbs_u_real.size(0)
                    corr_lb = (lbs_u_real == lbs_u)
                    loss_u_real = F.cross_entropy(pred_u_s, lbs_u_real)
            else:
                loss_u = torch.tensor(0)
                loss_u_real = torch.tensor(0)
                corr_lb = torch.tensor(0)
                valid_num = 0

            loss = loss_l + cfg.lam_u * loss_u + 0.1 * T_MI_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update_params()
            self.lr_schdlr.step()

            Loss.update(loss.item())
            Loss_L.update(loss_l.item())
            Loss_U.update(loss_u.item())
            Loss_U_Real.update(loss_u_real.item())
            Loss_MI.update(T_MI_loss.item())
            Correct_Num.update(corr_lb.sum().item())
            Valid_Num.update(valid_num)

            if (it+1) % 256 == 0:
                self.i_tb += 1
                self.writer.add_scalar('loss_u', Loss_U.avg, self.i_tb)
                self.writer.add_scalar('loss_MI', Loss_MI.avg, self.i_tb)
                ed = time.time()
                t = ed -st
                lr_log = [pg['lr'] for pg in self.optimizer.param_groups]
                lr_log = sum(lr_log) / len(lr_log)
                msg = ', '.join([
                    '     [iter: {}',
                    'loss: {:.3f}',
                    'loss_l: {:.4f}',
                    'loss_u: {:.4f}',
                    'loss_u_real: {:.4f}',
                    'loss_MI: {:.4f}',
                    'correct: {}/{}',
                    'lr: {:.4f}',
                    'time: {:.2f}]',
                ]).format(
                    it+1, Loss.avg, Loss_L.avg, Loss_U.avg,Loss_U_Real.avg, Loss_MI.avg,
                    int(Correct_Num.avg), int(Valid_Num.avg), lr_log, t
                )
                st = ed
                print(msg)

        self.ema.update_buffer()
        self.writer.add_scalar('acc_overall', Correct_Num.sum/(cfg.n_imgs_per_epoch*cfg.mu), self.epoch+1)
        self.writer.add_scalar('acc_in_labeled', Correct_Num.sum/(Valid_Num.sum+1e-10), self.epoch+1)

    def evaluate(self,ema):
        ema.apply_shadow()
        ema.model.eval()
        ema.model.cuda()

        matches = []
        for ims, lbs in self.val_loader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            with torch.no_grad():
                __,preds = ema.model(ims, mode='val')
                scores = torch.softmax(preds, dim=1)
                _, preds = torch.max(scores, dim=1)
                match = lbs == preds
                matches.append(match)
        matches = torch.cat(matches, dim=0).float()
        acc = torch.mean(matches)
        ema.restore()

        self.writer.add_scalar('val_acc', acc, self.epoch)
        self.train_record = update_model(ema.model, self.optimizer,self.lr_schdlr,self.epoch, self.i_tb,self.exp_path,self.exp_name,acc, self.train_record)
        print_summary(cfg.exp_name,acc, self.train_record)



if __name__ == '__main__':

    import os
    # ------------prepare enviroment------------
    seed = cfg.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    pwd = os.path.split(os.path.realpath(__file__))[0]
    trainer = Trainer(pwd)
    trainer.forward()

