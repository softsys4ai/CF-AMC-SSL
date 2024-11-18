############
## Import ##
############
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from model.model import encoder_simclr
from dataset.datasets import load_dataset_simclr
from tqdm import tqdm
import torch
from lars import LARSWrapper
import torch.optim.lr_scheduler as lr_scheduler


######################
## Parsing Argument ##
######################
import argparse
parser = argparse.ArgumentParser(description='Unsupervised Learning')
parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SimCLR', 'SupCon'], help='contrastive learning methods')
parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
parser.add_argument('--arch', type=str, default="resnet18-cifar",
                    help='network architecture (default: resnet18-cifar)')
parser.add_argument('--bs', type=int, default=512,
                    help='batch size (default: 100)')
parser.add_argument('--lr', type=float, default=0.3,
                    help='learning rate (default: 0.3)')        
parser.add_argument('--msg', type=str, default="NONE",
                    help='additional message for description (default: NONE)')     
parser.add_argument('--dir', type=str, default="SimCLR-AdvTraining",
                    help='directory name (default: SimCLR-AdvTraining)')     
parser.add_argument('--data', type=str, default="cifar100",
                    help='data (default: cifar10)')          
parser.add_argument('--epoch', type=int, default=500,
                    help='max number of epochs to finish (default: 30)')  
parser.add_argument('--scale_min', type=float, default=0.08, 
                    help='Minimum scale for resizing')

parser.add_argument('--scale_max', type=float, default=1, 
                    help='Maximum scale for resizing')

parser.add_argument('--ratio_min', type=float, default=0.75, 
                    help='Minimum aspect ratio')

parser.add_argument('--ratio_max', type=float, default=1.333333333, 
                    help='Maximum aspect ratio') 

args = parser.parse_args()

print(args)

print('hi')
dir_name = f"./logs/{args.dir}/{args.method}_bs{args.bs}_{args.msg}"
print(dir_name)

#####################
## Helper Function ##
#####################
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


######################
## Prepare Training ##
######################
torch.multiprocessing.set_sharing_strategy('file_system')

train_dataset = load_dataset_simclr(args, train=True)
dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=16)


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    
    
net = encoder_simclr(arch = args.arch)
net = nn.DataParallel(net)
net.cuda()


opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)
opt = LARSWrapper(opt,eta=0.005,clip=True,exclude_bias_n_norm=True,)

num_converge = (50000//args.bs)*args.epoch   
scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=num_converge, eta_min=0,last_epoch=-1)

# Loss

criterion = SupConLoss(temperature=args.temp)

def pgd_linf(model, X, epsilon, alpha, num_iter,criterion,labels, method):    
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        z = model(X)
        z_adv = model(X+delta)
        features = torch.cat([z.unsqueeze(1), z_adv.unsqueeze(1)], dim=1)
        if method == 'SupCon':
            loss = criterion(features, labels).to(device)
        elif method == 'SimCLR':
            loss = criterion(features).to(device)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()

##############
## Training ##
##############
# Loading
# save_dict = torch.load('')
# net.load_state_dict(save_dict,strict=False)
# net.cuda()
net.train()
def main():
    for epoch in range(args.epoch): 
        totalStep = len(dataloader)           
        for step, (data, label) in tqdm(enumerate(dataloader)):
            net.zero_grad()
            opt.zero_grad()
            x1 = data[0].to(device)
            x2 = data[1].to(device)
            label = label.to(device)
            delta = pgd_linf(net, x2, 8/255, 1e-2, 5, criterion,label,args.method)
            x_adv = (x2 + delta)
            # Forward pass
            z1 = net(x1)
            z2 = net(x_adv)

            features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
            if args.method == 'SupCon':
                loss = criterion(features, label).to(device)
            elif args.method == 'SimCLR':
                loss = criterion(features).to(device)
          
            loss.backward()
            opt.step()
            scheduler.step()
            
           
        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, args.epoch, step+1, totalStep, loss.item()),flush=True)
        if (epoch+1) % 100 == 1:
            print('hi')
            model_dir = f"{dir_name}/save_models_{args.data}/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(net.state_dict(), f"{model_dir}{epoch}.pt")
        
        
    


if __name__ == '__main__':
    main()


