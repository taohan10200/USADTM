import torch
from misc.DTM import Cluster

class LabelGuessor(object):

    def __init__(self, args):
        self.label_generator = Cluster(args.n_labeled // args.n_classes * 10, num_class=args.n_classes,
                                  feature_len=128).cuda()

        self.dataset = args.dataset
        self.args = args
        self.start_flag = 0
    def init_for_add_sample(self,epoch,start_epoch):
        if self.dataset == 'CIFAR10':
            if self.args.n_labeled ==40:
                num_add_sample =min(int((epoch-start_epoch) * 0.1)+1, 4)
            if self.args.n_labeled ==250:
                num_add_sample =min((epoch-start_epoch) * 2, 50)
            if self.args.n_labeled ==4000:
                num_add_sample =min((epoch-start_epoch) * 10, 800)
        if self.dataset == 'CIFAR100':
            if self.args.n_labeled == 400:
                num_add_sample = min(int((epoch-start_epoch) * 0.01)+1, 4)
            if self.args.n_labeled == 2500:
                num_add_sample = min(int((epoch-start_epoch) * 0.2)+1, 50)
            if self.args.n_labeled == 10000:
                num_add_sample = min((epoch-start_epoch) * 2, 200)
        if self.dataset == 'SVHN':
            if self.args.n_labeled == 40:
                num_add_sample = min(int((epoch-start_epoch) * 0.1)+1, 4)
            if self.args.n_labeled == 250:
                num_add_sample = min(int((epoch-start_epoch) * 1)+1, 50)
            if self.args.n_labeled == 1000:
                num_add_sample = min((epoch-start_epoch) * 2, 200)
        if self.dataset == 'STL10':
            if self.args.n_labeled == 40:
                num_add_sample = min(int((epoch-start_epoch) * 0.1)+1, 4)
            if self.args.n_labeled == 250:
                num_add_sample = min((epoch-start_epoch) * 1, 50)
            if self.args.n_labeled == 1000:
                num_add_sample = min((epoch-start_epoch) * 2, 200)
        self.label_generator.init_(num_add_sample, (num_add_sample+self.args.n_labeled//self.args.n_classes))


    def __call__(self, model, img_l_weak,ims_u_weak,lbs_l,unlabeled_index=None):
        org_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        is_train = model.training
        with torch.no_grad():
            input = torch.cat([img_l_weak, ims_u_weak], dim=0).detach()
            f_l, f_u, pred_l_w,pred_u_w = model(input)

            self.label_generator.add_sample(f_l.detach(), lbs_l.detach())

            if self.start_flag == 0:
                count = 0
                for i in range (self.args.n_classes):
                    if self.label_generator.class_pool.num_imgs[i]>10:
                        count+=1
                    if count==self.args.n_classes:
                        self.start_flag=1
                pseudo = torch.zeros(f_u.size(0)).fill_(-1)
                idx = pseudo > -1
                lbs = pseudo[idx]

                model.load_state_dict(org_state)
                if is_train:
                    model.train()
                else:
                    model.eval()
                return lbs.detach(), idx
            else:
                pseudo = self.label_generator.forward(f_u.detach(), pred_u_w.detach(),unlabeled_index).long().cuda()
                idx = pseudo > -1
                lbs = pseudo[idx]

                model.load_state_dict(org_state)
                if is_train:
                    model.train()
                else:
                    model.eval()
                return lbs.detach(), idx
