import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import clip_w_local
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True



def robust_loader(path):
    from torchvision.datasets.folder import default_loader
    try:
        return default_loader(path)
    except (OSError, UnidentifiedImageError):
        print(f"Skipping corrupt image: {path}")
        return Image.new('RGB', (224, 224))


def set_model_clip(args):
    model, _ = clip_w_local.load(args.CLIP_ckpt)

    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return model, val_preprocess


def set_val_loader(args, preprocess=None):
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "imagenet":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet/images/val'), transform=preprocess, loader=robust_loader),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "imagenet100":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet100/images/val'), transform=preprocess, loader=robust_loader),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "imagenet10":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet10/images/val'), transform=preprocess, loader=robust_loader),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "imagenet20":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet20/images/val'), transform=preprocess, loader=robust_loader),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError
    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess=None):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'iNaturalist'), transform=preprocess, loader=robust_loader)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'SUN'), transform=preprocess, loader=robust_loader)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Places'), transform=preprocess, loader=robust_loader)
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Texture', 'images'),
                                          transform=preprocess, loader=robust_loader)
    elif out_dataset == 'imagenet20':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet20', 'images/val'),
                                          transform=preprocess, loader=robust_loader)
    elif out_dataset == 'imagenet10':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet10', 'images/val'),
                                          transform=preprocess, loader=robust_loader)
    elif out_dataset == 'imagenet100':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet100', 'images/val'),
                                          transform=preprocess, loader=robust_loader)
    elif out_dataset == 'SSB_hard':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'ssb_hard'), transform=preprocess, loader=robust_loader)
    elif out_dataset == 'NINCO':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'NINCO'), transform=preprocess, loader=robust_loader)
    elif out_dataset == 'OpenImage-O':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'OpenImage-O'), transform=preprocess, loader=robust_loader)
    elif out_dataset == 'ImageNet-O':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'ImageNet-O'), transform=preprocess, loader=robust_loader)
    elif out_dataset == 'ImageNet1k-OOD':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet10k', 'images'), transform=preprocess, loader=robust_loader)
        
    else:
        raise NotImplementedError(f'Out dataset {out_dataset} not implemented yet!')
    
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
    return testloaderOut
