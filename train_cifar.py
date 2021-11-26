import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils.utils as utils
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from model.wideresnet import WideResNet
from model.preact_resnet import PreActResNet18
from model.discriminator import *
import time


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-d', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='beta')
    parser.add_argument('--beta2', default=0.99, type=float,
                        help='beta')
    parser.add_argument('--lambda1', default=0., type=float,
                        help='')
    parser.add_argument('--lambda2', default=1., type=float,
                        help='adv_loss')
    parser.add_argument('--lambda3', default=0.001, type=float,
                        help='faat')
    parser.add_argument('--lambda4', default=0., type=float,
                        help='trades loss')
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='perturbation')
    parser.add_argument('--num-steps', default=10, type=int,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.007, type=float,
                        help='perturb step size')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-dir', required=True,
                        help='directory of model for saving checkpoint')
    parser.add_argument('--data-dir', default='../DATA',
                        help='directory of dataset')
    parser.add_argument('--milestones', type=int, nargs='+', default=[55, 75, 90], help='milestones')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'],
                        help='dataset, cifar10 or cifar100')
    parser.add_argument('--arch', default='wideresnet34', choices=['preactresnet18', 'wideresnet34'],
                        help='')
    parser.add_argument('--resume', default='', type=str,
                        help='directory of model for saving checkpoint')
    parser.add_argument('--vis-freq', default=50, type=int, metavar='N',
                        help='')
    parser.add_argument('--eval-freq', default=10, type=int, metavar='N',
                        help='eval frequency')
    parser.add_argument('--save-freq', default=5, type=int, metavar='N',
                        help='save frequency')
    return parser.parse_args()


def train():
    # adversary = LinfPGDAttack(
    #     netG, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon,
    #     nb_iter=args.num_steps, eps_iter=args.step_size, rand_init=True, clip_min=0.0,
    #     clip_max=1.0, targeted=False)
    global is_best, best_rob_acc, start_epoch, total_time

    cross_entropy = nn.CrossEntropyLoss()
    writer = SummaryWriter(os.path.join(args.model_dir, 'log'))
    train_loader, test_loader = utils.get_loaders(args.data_dir, args.batch_size, args.dataset)

    for epoch in range(start_epoch, args.epochs + 1):
        total_err_g = 0
        total_err_d = 0
        total_err_g_d_loss = 0
        total_nat_ent_loss = 0
        total_adv_ent_loss = 0
        total_trades_loss = 0
        is_best = False
        for i, (X, y) in enumerate(train_loader, 0):
            start_time = time.time()
            netG.train()
            X, y = X.to(device), y.to(device)
            X_adv = utils.attack_pgd(X, y, netG, attack_iters=args.num_steps)

            ############################
            # (1) Update D
            ###########################
            netD.zero_grad()

            X_pred = netG(X)
            X_feat = netG.feat

            X_adv_pred = netG(X_adv)
            X_adv_feat = netG.feat

            output = netD(X_feat.detach())
            errD_real = utils.soft_label_cross_entropy(output, torch.cat((torch.zeros_like(y), y), dim=0),
                                                       n_classes=n_classes)

            output = netD(X_adv_feat.detach())
            errD_fake = utils.soft_label_cross_entropy(output, torch.cat((y, torch.zeros_like(y)), dim=0),
                                                       n_classes=n_classes)

            errD = 0.5 * errD_real + 0.5 * errD_fake
            errD.backward()
            optimizerD.step()

            ############################
            # (2) Update G
            ###########################
            netG.zero_grad()
            output = netD(X_feat)

            nat_ent_loss = args.lambda1 * cross_entropy(X_pred, y)
            adv_ent_loss = args.lambda2 * cross_entropy(X_adv_pred, y)
            err_g_d_loss = args.lambda3 * utils.soft_label_cross_entropy(output,
                                                                         torch.cat((y, torch.zeros_like(y)), dim=0),
                                                                         n_classes=n_classes)
            trades_loss = args.lambda4 * utils.kl_div(X_adv_pred, X_pred)
            errG = nat_ent_loss + adv_ent_loss + err_g_d_loss + trades_loss

            errG.backward()
            optimizerG.step()

            oprint(time.time() - start_time)
            total_time += time.time() - start_time

            # Output training stats
            if i % args.vis_freq == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    '\terr_g_d_loss: %.4f\tnat_ent_loss: %.4f\tadv_ent_loss: %.4f\tloss_mart: %.4f'
                    % (epoch, args.epochs, i, len(train_loader), errD.item(), errG.item(),
                       err_g_d_loss.item(), nat_ent_loss.item(), adv_ent_loss.item(), trades_loss.item()))

            total_err_g_d_loss += err_g_d_loss.item()
            total_nat_ent_loss += nat_ent_loss.item()
            total_adv_ent_loss += adv_ent_loss.item()
            total_trades_loss += trades_loss.item()

            total_err_g += errG.item()
            total_err_d += errD.item()

        _, nat_acc = utils.evaluate_standard(test_loader, netG)

        if epoch % args.eval_freq == 0:
            rob_acc = utils.evaluate_pgd(test_loader, netG)
        else:
            rob_acc = utils.evaluate_pgd(test_loader, netG, n_eval=512)

        if rob_acc > best_rob_acc:
            best_rob_acc = rob_acc
            is_best = True

        if is_best or epoch % args.save_freq == 0:
            utils.save_checkpoint(netG.state_dict(), is_best=is_best, checkpoint=args.model_dir,
                                  filename="model-{}.pt".format(epoch))

        print('[%d/%d]\tnatural acc: %.4f\trobust acc: %.4f\td_loss: %.4f\tg_loss:%.4f'
              '\terr_g_d_loss: %.4f\tnat_ent_loss: %.4f\tadv_ent_loss: %.4f' %
              (epoch, args.epochs, nat_acc, rob_acc, total_err_d / len(train_loader), total_err_g / len(train_loader),
               total_err_g_d_loss / len(train_loader), total_nat_ent_loss / len(train_loader),
               total_adv_ent_loss / len(train_loader)))

        writer.add_scalars('Loss', {'total_err_d': total_err_d / len(train_loader),
                                    'total_err_g': total_err_g / len(train_loader),
                                    'total_err_g_d_loss': total_err_g_d_loss / len(train_loader),
                                    'total_nat_ent_loss': total_nat_ent_loss / len(train_loader),
                                    'total_adv_ent_loss': total_adv_ent_loss / len(train_loader),
                                    'total_trades_loss': total_trades_loss / len(train_loader)
                                    }, epoch)

        writer.add_scalars('Accuracy', {
            'natural_acc': nat_acc,
            'robust_acc': rob_acc}, epoch)
        writer.add_scalar('learning rate', optimizerG.state_dict()['param_groups'][0]['lr'], epoch)

        schedulerG.step()

        torch.save({'netG': netG.state_dict(),
                    'netD': netD.state_dict(),
                    'schedulerG': schedulerG.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'epoch': epoch,
                    'is_best': is_best,
                    'best_rob_acc': best_rob_acc
                    }, os.path.join(args.model_dir, 'last.pt'))

        writer.close()


args = parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'cifar10':
    n_classes = 10
    stride = 1
elif args.dataset == 'cifar100':
    n_classes = 100
    stride = 1
elif args.dataset == 'tinyimagenet':
    n_classes = 200
    stride = 2
else:
    raise ValueError('no such dataset!')

if args.arch == 'preactresnet18':
    netG = PreActResNet18(num_classes=n_classes, stride=stride).to(device)
elif args.arch == 'wideresnet34':
    netG = WideResNet(device=device, depth=34, num_classes=n_classes, stride=stride).to(device)
else:
    raise KeyError("not found arch {}".format(args.arch))

# logging
logger = utils.init_logger(args.model_dir, filename='log.txt')
logger.info(args)
oprint = print
print = logger.info

netD = PixelDiscriminator(n_classes=n_classes).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
optimizerG = optim.SGD(netG.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=args.milestones, gamma=0.1)

start_epoch = 1
is_best = False
best_rob_acc = 0

if args.resume:
    state_dict = torch.load(args.resume)
    netG.load_state_dict(state_dict['netG'])
    netD.load_state_dict(state_dict['netD'])
    optimizerG.load_state_dict(state_dict['optimizerG'])
    optimizerD.load_state_dict(state_dict['optimizerD'])
    schedulerG.load_state_dict(state_dict['schedulerG'])
    is_best = state_dict['is_best']
    best_rob_acc = state_dict['best_rob_acc']
    start_epoch = state_dict['epoch'] + 1

    print("resume from epoch: {}, best_rob_acc: {:.4f}, is_best: {}".format(start_epoch, best_rob_acc, is_best))

total_time = 0
train()
print(total_time)
logger.info("total time: {:.4f}".format(total_time / 60))
