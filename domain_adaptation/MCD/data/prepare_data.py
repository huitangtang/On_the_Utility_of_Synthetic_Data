import os
import shutil
import torch
import scipy.io as scio
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.folder_new import ImageFolder_new


def generate_dataloader(args):
    # Data loading code
    traindir_source = args.source_data_path
    traindir_target = args.target_data_path
    valdir = args.test_data_path
    if not os.path.isdir(traindir_source):
        raise ValueError('Null path of source training data!!!')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    source_train_dataset = datasets.ImageFolder(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size_s, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
    )

    source_val_dataset = ImageFolder_new(
        traindir_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    source_val_loader = torch.utils.data.DataLoader(
        source_val_dataset, batch_size=args.batch_size_s, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )
    
    target_train_dataset = datasets.ImageFolder(
        traindir_target,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size_t, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True, sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        ImageFolder_new(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size_t, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    ## return Source (random crop), Source (Center crop), Target (random crop), Traget (Center crop)
    return source_train_loader, source_val_loader, target_train_loader, val_loader

