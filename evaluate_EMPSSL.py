############
## Import ##
############
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import encoder
from dataset.datasets import load_dataset
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from func import linear_train
import torchvision.transforms as transforms
from dataset.aug import ContrastiveLearningViewGenerator
import torchvision
import torchattacks
######################
## Parsing Argument #
######################
import argparse
parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--test_patches', type=int, default=4,
                    help='number of patches used in testing (default: 128)')  

parser.add_argument('--data', type=str, default="cifar10",
                    help='dataset (default: cifar10)')  

parser.add_argument('--arch', type=str, default="resnet18-cifar",
                    help='network architecture (default: resnet18-cifar)')

parser.add_argument('--lr', type=float, default=0.03,
                    help='learning rate for linear eval (default: 0.03)')       

parser.add_argument('--linear', type=bool, default=True,
                    help='use linear eval or not')

parser.add_argument('--model_path', type=str, default="",
                    help='model directory for eval')

parser.add_argument('--scale_min', type=float, default=0.08, 
                    help='Minimum scale for resizing')

parser.add_argument('--scale_max', type=float, default=1, 
                    help='Maximum scale for resizing')

parser.add_argument('--ratio_min', type=float, default=0.75, 
                    help='Minimum aspect ratio')

parser.add_argument('--ratio_max', type=float, default=1.3333333333333, 
                    help='Maximum aspect ratio')

parser.add_argument('--type', type=str, default="crop",
                    help='crop vs. patch')

parser.add_argument('--epochs', type=int, default=2,
                    help='max number of epochs to finish')  

parser.add_argument('--bs_centralcrop_train', type=int, default =256,
                    help='batchSize for training central_crop') 

parser.add_argument('--bs_centralcrop_test', type=int, default =100,
                    help='batchSize for testing central_crop') 


parser.add_argument('--bs_patch_train', type=int, default = 100,
                    help=' batchSize for training n patch (crop) classifier ') 

parser.add_argument('--bs_patch_test', type=int, default = 16,
                    help=' batchSize for testing n patch (crop) classifier') 

parser.add_argument('--alpha', type=float, default = 1e-2, 
                    help='movement multiplier per iteration in adversarial examples')

parser.add_argument('--iter', type=int, default = 2, 
                    help='number of iterations for generating adversarial Examples')

parser.add_argument('--hidden_units', type=int, default = 4096, 
                    help='number of representations')



            
args = parser.parse_args()

print("Running with test_patches = " + str(args.test_patches) + "model_path = " + args.model_path + "/type = " + args.type)

######################
## Testing Accuracy ##
######################
test_patches = args.test_patches

if args.data=='cifar10':
     args.num_class = 10
else:
     args.num_class = 100

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size




def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


def train_Eval(net, train_loader):
    
    train_z_full_list, train_y_list, test_z_full_list, test_y_list = [], [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(train_loader):

            x = torch.cat(x, dim = 0)
            
            z_proj, z_pre = net(x, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.detach().cpu()
            train_z_full_list.append(z_pre)
            train_y_list.append(y)
                
    train_features_full, train_labels = torch.cat(train_z_full_list,dim=0), torch.cat(train_y_list,dim=0)
   
        
 
    LL = linear_train(train_features_full, train_labels, lr=args.lr, num_classes = args.num_class)
    return LL
    

    
def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)



#Get Dataset
if args.data == "imagenet100" or args.data == "imagenet":
        
    memory_dataset = load_dataset(args, args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=args.bs_patch_train, shuffle=True, drop_last=True,num_workers=8)

    test_data = load_dataset(args, args.data, train=False, num_patch = test_patches)
    test_loader = DataLoader(test_data, batch_size=args.bs_patch_train, shuffle=True, num_workers=8)

else:
  
    memory_dataset = load_dataset(args, args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=args.bs_patch_train, shuffle=True, drop_last=True,num_workers=8)

   

# Load Model and Checkpoint
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
net = encoder(arch = args.arch)
net = nn.DataParallel(net)

save_dict = torch.load(args.model_path)
net.load_state_dict(save_dict,strict=False)
net.cuda()
net.eval()
LL = train_Eval(net, memory_loader)
LL.cuda()



print('################### Test based on n patch(crop) linear classifer on clean data #############')
testTransform = ContrastiveLearningViewGenerator(num_patch=args.test_patches, scale_min = args.scale_min, scale_max = args.scale_max, ratio_min = args.ratio_min, ratio_max = args.ratio_max)
if args.data == 'cifar10':
    testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform, download=True)
elif args.data == 'cifar100':
    testDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=False, transform=testTransform, download=True) 

batchSize = args.bs_patch_test
testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )

def test_n_patch_cls():
    net.eval()
    LL.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            
            X = torch.cat(X, dim = 0).to(device)
            labels = labels.to(device)
            z_proj, z_pre = net(X, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            yp = LL(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)

# test_n_patch_cls()



def pgd_linf_n_patch(BE,my_LL, X, y, epsilon, alpha, num_iter):
    
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        _,z_pre =  BE(X+delta, is_test=True)
        z_pre = chunk_avg(z_pre, test_patches)
        z_pre = z_pre.to(device)
        yp = my_LL(z_pre).to(device)
        loss = nn.CrossEntropyLoss()(yp, y)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()


# def pgd_linf_n_patch_iid(BE,my_LL, X, y, epsilon, alpha, num_iter):
#     """ Construct FGSM adversarial examples on the examples X"""
#     delta = torch.zeros_like(X, requires_grad=True)
#     y = torch.repeat_interleave(y,test_patches)
#     for t in range(num_iter):
#         _,z_pre =  BE(X+delta, is_test=True)
#         # z_pre = chunk_avg(z_pre, test_patches)
#         z_pre = z_pre.to(device)
#         yp = my_LL(z_pre).to(device)
#         loss = nn.CrossEntropyLoss()(yp, y)
#         loss.backward()
#         with torch.no_grad():
#             delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#             delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
#         delta.grad.zero_()
#     return delta.detach()

print('################### Test based on n patch(crop) linear classifer on adversarial examples generated by n patch#############')

def test_n_patch_cls_adv(eps):
    net.eval()
    LL.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = torch.cat(X, dim = 0).to(device)
            labels = labels.to(device)
            delta = pgd_linf_n_patch(net,LL, X, labels, eps, args.alpha, args.iter)
            _,z_pre =  net(X+delta, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.to(device)
            yp = LL(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)

# test_n_patch_cls_adv(4/255)
# test_n_patch_cls_adv(8/255)
# test_n_patch_cls_adv(16/255)

# print('###################Test based on n patch(crop) linear classifer on iid adversarial examples#############')
# def test_n_patch_cls_adv_iid (eps):
#     net.eval()
#     LL.eval()
#     total_acc_test = 0
#     for i, (X, labels) in tqdm(enumerate(testLoader)):
#             X = torch.cat(X, dim = 0).to(device)
#             labels = labels.to(device)
#             delta = pgd_linf_n_patch_iid(net,LL, X, labels, eps, args.alpha, args.iter)
#             _,z_pre =  net(X+delta, is_test=True)
#             z_pre = chunk_avg(z_pre, test_patches)
#             z_pre = z_pre.to(device)
#             yp = LL(z_pre).to(device)
#             total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
#     print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
#     return total_acc_test / len(testLoader.dataset)

# test_n_patch_cls_adv_iid(4/255)
# test_n_patch_cls_adv_iid(8/255)
# test_n_patch_cls_adv_iid(16/255)




print('###################Train based on central crop classifier#############')

trainEvalTransform = transforms.Compose([transforms.ToTensor()])
if args.data == 'cifar10':
    trainEvalDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=True, transform=trainEvalTransform, download=True)
elif args.data=='cifar100':
    trainEvalDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=True, transform=trainEvalTransform, download=True)

batchSize = args.bs_centralcrop_train
trainEvalLoader      = torch.utils.data.DataLoader(dataset=trainEvalDataset ,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )

num_class = args.num_class
LL2 = nn.Linear(args.hidden_units,num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(LL2.parameters(), lr=3e-4)
numEpochs = args.epochs

def train_central_crop_cls():
    totalStep = len(trainEvalLoader)
    net.eval()
    LL2.train()
    for epoch in range(numEpochs):
        for i, (X, labels) in enumerate(trainEvalLoader):
            X = X.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                _,z_pre =  net(X, is_test=True)
                z_pre = z_pre.to(device)
            yp = LL2(z_pre)
            loss = criterion(yp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, numEpochs, i+1, totalStep, loss.item()),flush=True)
    # PATH = './CIFAR100_EvalNet_Epoch='+str(numEpochs)+'_BatchSize='+str(batchSize)+'.pt'
    # torch.save(LL2.state_dict(), PATH)
    
train_central_crop_cls()  

def pgd_linf_end2end(BE,my_LL, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        _,z_pre =  BE(X+delta, is_test=True)
        z_pre = z_pre.to(device)
        yp = my_LL(z_pre).to(device)
        loss = nn.CrossEntropyLoss()(yp, y)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()

def train_robust_central_crop_cls():
    totalStep = len(trainEvalLoader)
    net.eval()
    LL2.train()
    for epoch in range(numEpochs):
        for i, (X, labels) in tqdm(enumerate(trainEvalLoader)):
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end2end(net, LL2, X, labels, 8/255, args.alpha, 5)
            X_adv = (X + delta)
            with torch.no_grad():
                _,z_pre =  net(X_adv, is_test=True)
                z_pre = z_pre.to(device)
            yp = LL2(z_pre)
            loss = criterion(yp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_robust_central_crop_cls()


print('###################Test based on central crop classifier on clean data#############')

testTransform = transforms.Compose([
        transforms.ToTensor()])

if args.data == 'cifar10':
    testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform, download=True)
elif args.data == 'cifar100':
    testDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=False, transform=testTransform, download=True)

batchSize = args.bs_centralcrop_test

testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )


def test_central_crop_cls():
    net.eval()
    LL2.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = X.to(device)
            labels = labels.to(device)
            _,z_pre =  net(X, is_test=True)
            z_pre = z_pre.to(device)
            yp = LL2(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)

train_central_crop_cls()


print('###################Test based on central crop classifier on adversarial examples#############')



def test_central_crop_cls_adv(epst):
    totalStep = len(testLoader)
    net.eval()
    LL2.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end2end(net, LL2, X, labels, epst, args.alpha, args.iter)
            X_adv = (X + delta)
            _,z_pre =  net(X_adv, is_test=True)
            z_pre = z_pre.to(device)
            yp = LL2(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)


test_central_crop_cls_adv(4/255)
# test_central_crop_cls_adv(8/255)
# test_central_crop_cls_adv(16/255)


print('###################Test based on central crop classifier against Auto-attack#############')

class EncoderWithHead(nn.Module):
    def __init__(self, encoder, head):
        super(EncoderWithHead, self).__init__()
        self.encoder        = encoder
        self.head = head    
    def forward(self, x):
        _,z_pre =  self.encoder(x, is_test=True)
        z_pre = z_pre.to(device)
        yp = self.head(z_pre).to(device)
        return yp
    
EvalNet    = EncoderWithHead(net, LL2)

def test_central_crop_cls_Autoattack(epst):
    totalStep = len(testLoader)
    net.eval()
    LL2.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = X.to(device)
            labels = labels.to(device)
            attack = torchattacks.AutoAttack(EvalNet, norm='Linf', eps=epst, version='standard', n_classes=num_class, seed=None, verbose=False)
            X_adv = attack(X,labels)
            yp = EvalNet(X_adv)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)


test_central_crop_cls_Autoattack(4/255)
# test_central_crop_cls_Autoattack(8/255)
# test_central_crop_cls_Autoattack(16/255)

print('###################Test based on n patch(crop) classifier on adversarial examples #############')



def test_n_patch_cls_adv_end2end(epst):
    totalStep = len(testLoader)
    net.eval()
    LL.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end2end(net, LL, X, labels, epst, args.alpha, args.iter)
            X_adv = (X + delta)
            _,z_pre =  net(X_adv, is_test=True)
            z_pre = z_pre.to(device)
            yp = LL(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)


test_n_patch_cls_adv_end2end(2/255)
# test_n_patch_cls_adv_end2end(4/255)
# test_n_patch_cls_adv_end2end(8/255)
# test_n_patch_cls_adv_end2end(16/255)

