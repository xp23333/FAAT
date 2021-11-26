import os
import logging
import shutil
import torch
import torch.nn.functional as F
import foolbox as fb
import numpy as np

from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)


cifar10_mean = (0., 0., 0.)
cifar10_std = (1., 1., 1.)

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def get_loaders(dir_, batch_size, dataset='cifar10', type=None):
    if dataset == 'tinyimagenet':
        image_size = (3, 64, 64)
    else:
        image_size = (3, 32, 32)

    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size[1], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    if type == 'ccg':
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        train_transform.transforms.append(CutoutDefault(int(32 / 2)))

    if type == 'ccg+r':
        train_transform = MultiDataTransform(train_transform)

    num_workers = 8
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=False)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=False)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'tinyimagenet':
        # image_size = (3, 64, 64)
        # n_classes = 200
        train_dir = os.path.join(dir_, 'tiny-imagenet-200', 'train')
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        test_dir = os.path.join(dir_, 'tiny-imagenet-200', 'val')
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    else:
        raise KeyError('no such dataset')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def evaluate_standard(test_loader, model):
    was_training = model.training
    model.eval()
    test_loss = 0
    test_acc = 0
    n = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    model.train(was_training)
    return test_loss / n, test_acc / n


def attack_pgd(X, y, model, attack_iters=10, step_size=2 / 255., epsilon=8 / 255.):
    was_training = model.training
    model.eval()
    preprocessing = dict(mean=cifar10_mean, std=cifar10_std, axis=-3)

    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    pgd_attack = fb.attacks.PGD(abs_stepsize=step_size, steps=attack_iters, random_start=True)

    raw, clipped, is_adv = pgd_attack(fmodel, X, y, epsilons=epsilon)

    model.train(was_training)
    return clipped


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def CW_loss(logits, target, num_class=10):
    label_masks = F.one_hot(target, num_class)
    correct_logit = (label_masks * logits).sum(1)
    wrong_logit = (1 - label_masks) * logits - 1e4 * label_masks
    wrong_logit, _ = wrong_logit.max(1)
    loss = F.relu(correct_logit - wrong_logit + 50)
    loss = -loss.sum()
    return loss


def attack_pgd_cw(model, X, y, epsilon, alpha, attack_iters, restarts, num_classes):
    if not isinstance(epsilon, torch.Tensor):
        epsilon = epsilon / torch.tensor((1., 1., 1.)).view(3, 1, 1).cuda()
        alpha = alpha / torch.tensor((1., 1., 1.)).view(3, 1, 1).cuda()

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y, num_class=num_classes)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters=10, step_size=0.007, epsilon=0.031, n_eval=10000):
    was_training = model.training
    model.eval()
    bounds = (0, 1)
    preprocessing = dict(mean=cifar10_mean, std=cifar10_std, axis=-3)
    fmodel = fb.models.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    pgd_attack = fb.attacks.PGD(abs_stepsize=step_size, steps=attack_iters, random_start=True)

    total_adv = 0
    total = 0

    for i, (X, y) in enumerate(tqdm(test_loader)):
        X, y = X.cuda(), y.cuda()
        raw, clipped, is_adv = pgd_attack(fmodel, X, y, epsilons=epsilon)

        total_adv += is_adv.sum(-1).item()
        total += is_adv.shape[0]

        if total >= n_eval:
            break

    robust_accuracy = 1 - total_adv / total

    model.train(was_training)
    return robust_accuracy


def evaluate_pgd_0(test_loader, model, attack_iters=10, step_size=0.007, epsilon=0.031, n_eval=10000):
    from advertorch.attacks import LinfPGDAttack
    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
        nb_iter=attack_iters, eps_iter=step_size, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    was_training = model.training
    model.eval()
    test_advloss = 0
    advcorrect = 0
    n = 0
    for i, (X, y) in enumerate(tqdm(test_loader)):
        X, y = X.cuda(), y.cuda()
        adv = adversary.perturb(X, y)

        with torch.no_grad():
            output = model(adv)
        test_advloss += F.cross_entropy(output, y, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]

        advcorrect += pred.eq(y.view_as(pred)).sum().item()
        n += X.size(0)
        if n > n_eval:
            break

    model.train(was_training)
    return advcorrect / n


def save_checkpoint(state, is_best=False, checkpoint='checkpoint', filename='checkpoint.pt'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pt'))


def init_logger(out_dir, filename=None, level=logging.INFO):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if len(os.listdir(out_dir)) != 0:
        ans = input("out_dir is not empty. All data inside out_dir will be deleted. "
                    "Will you proceed [y/N]? ")
        if ans in ['y', 'Y']:
            shutil.rmtree(out_dir)
        else:
            pass

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if filename is not None:
        logfile = os.path.join(out_dir, filename)
    else:
        logfile = os.path.join(out_dir, 'log.txt')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=level,
        filename=logfile)

    return logging.getLogger(__name__)


def soft_label_cross_entropy(pred, target, n_classes=10, label_smooth=0.03, reduction='mean'):
    batch_size = pred.size(0)

    target = F.one_hot(target, n_classes)  # 转换成one-hot
    t1, t2 = torch.split(target, split_size_or_sections=target.size(0) // 2, dim=0)
    target = torch.cat((t1, t2), dim=1)

    target = torch.clamp(target.float(), min=label_smooth / (n_classes - 1), max=1.0 - label_smooth)

    logprobs = F.log_softmax(pred, dim=1)  # softmax + log
    loss = -1. * torch.sum(target * logprobs, dim=1)

    if reduction == 'mean':
        loss = loss.sum() / batch_size
    if reduction == 'sum':
        loss = loss.sum()

    return loss


def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), F.softmax(targets, dim=1),
                    reduction=reduction)
