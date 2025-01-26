import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
from matplotlib import pyplot as plt

class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       
class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            
class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 

class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic2018':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic2017':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        else:
            raise ValueError(f"Unsupported data_name: {data_name}")
            
    def __call__(self, data):
        img, mask = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, mask
    
def seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_optimizer(optimizer_opt, model):
    assert optimizer_opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD']

    if optimizer_opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = 0.01,
            rho = 0.9,
            eps = 1e-6,
            weight_decay = 0.05
        )
    elif optimizer_opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = 0.01,
            lr_decay = 0,
            eps = 1e-10,
            weight_decay = 0.05
        )
    elif optimizer_opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = 0.001,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0.0001,
            amsgrad = False
        )
    elif optimizer_opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = 0.0001,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 1e-2,
            amsgrad = False
        )
    elif optimizer_opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = 2e-3,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0
        )
    elif optimizer_opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = 0.01,
            lambd = 1e-4,
            alpha  = 0.75,
            t0 = 1e6,
            weight_decay = 0
        )
    elif optimizer_opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = 1e-2,
            momentum = 0,
            alpha = 0.99,
            eps = 1e-8,
            centered = False,
            weight_decay = 0
        )
    elif optimizer_opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = 1e-2,
            etas = (0.5, 1.2),
            step_sizes = (1e-6, 50)
        )
    elif optimizer_opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
            dampening = 0,
            nesterov = False
        )
    else:
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, pred, target):
        epsilon = 1e-5
        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + epsilon)/(pred_flat.sum(1) + target_flat.sum(1) + epsilon)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class Loss(nn.Module):
    def __init__(self, w1, w2):
        super(Loss, self).__init__()
        self.diceloss = DiceLoss()
        self.crossentropyloss = nn.CrossEntropyLoss()
        self.w1 = w1
        self.w2 = w2
    def forward(self, pred, target):
        diceloss = self.diceloss(pred, target)
        crossentropyloss = self.crossentropyloss(pred, target)
        
        loss = self.w1 * diceloss + self.w2 * crossentropyloss

        return loss

def get_scheduler(scheduler_option, optimizer):
    assert scheduler_option in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if scheduler_option == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = 300 // 5,
            gamma = 0.5,
            last_epoch = -1
        )
    elif scheduler_option == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 50,
            eta_min = 0.00001,
            last_epoch = -1
        )
        
    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255.0 if img.max() > 1.1 else img

    msk = msk.squeeze(0).cpu().numpy()
    msk_pred = msk_pred.squeeze(0).cpu().numpy()
    msk = msk[0]
    msk_pred = msk_pred[0]

    if datasets != 'retinal':
        msk = (msk > 0.5).astype(np.uint8)
        msk_pred = (msk_pred > threshold).astype(np.uint8)

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'

    plt.savefig(save_path + str(i))
    plt.close()