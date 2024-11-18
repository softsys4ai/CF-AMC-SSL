import os
import torchvision
import torchvision.transforms as transforms


class TwoAugmentedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_dataset_simclr(args, train=True, path="./data/"):
    """Loads a dataset for training and testing.
    
    Parameters:
        data_name (str): name of the dataset
        train (bool): load training set or not
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    data_name = args.data
    _name = data_name.lower()

    trainCLTransform = torchvision.transforms.Compose(
            [
                # transforms.RandomResizedCrop(32),
                transforms.RandomResizedCrop(32, scale=(args.scale_min,args.scale_max),ratio=(args.ratio_min,args.ratio_max)), # patch-simclr
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor()])

 
    
    transform = TwoAugmentedTransform(trainCLTransform)

    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "CIFAR10"), train=train, download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train, download=True, transform=transform)
        trainset.num_classes = 100        
    else:
        raise NameError("{} not found in trainset loader".format(_name))
    return trainset


def load_dataset(args, data_name, train=True, num_patch = 4, path="./data/"):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    
    Parameters:
        data_name (str): name of the dataset
        transform_name (torchvision.transform): name of transform to be applied (see aug.py)
        train (bool): load training set or not
        n_patch: number of patches (crops)
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    _name = data_name.lower()
    from .aug import ContrastiveLearningViewGenerator
      
    
    transform = ContrastiveLearningViewGenerator(num_patch = num_patch, scale_min = args.scale_min, scale_max = args.scale_max, ratio_min = args.ratio_min, ratio_max = args.ratio_max)
        
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "CIFAR10"), train=train, download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train, download=True, transform=transform)
        trainset.num_classes = 100     
    else:
        raise NameError("{} not found in trainset loader".format(_name))
    return trainset