from __future__ import print_function
import os
import torch
import argparse
import foolbox as fb
from torch.autograd import Variable
from model.wideresnet import WideResNet
import utils.utils as utils

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')

# import parameter
parser.add_argument('--data-dir', default='../DATA', help='dataset directory')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'], help='dataset')
parser.add_argument('--num-classes', default=200, type=int, choices=[10, 100, 200], help='the number of class')
parser.add_argument('--arch', default='wideresnet34', help='pretrained model')
parser.add_argument('--src-arch', default='preactresnet18', choices=['preactresnet18', 'wideresnet34'],
                    help='source model arch for black-box')
parser.add_argument('--attack', default='pgd',
                    choices=['fgsm', 'pgd', 'pgd_l1', 'pgd_l2', 'cw_l2', 'cw_li', 'autoattack'],
                    help='the adversary')
parser.add_argument('--vis', action='store_true', help='visualize the process of attack')
parser.add_argument('--num-steps', default=20, type=int, help='perturb number of steps')
parser.add_argument('--epsilon', default=0.031, type=float, help='perturbation')
parser.add_argument('--step-size', default=0.007, type=float, help='perturb step size')
parser.add_argument('--restarts', default=1, type=int, help='random restarts')

# white-box or black
parser.add_argument('--white-box-attack', action='store_true', help='whether perform white-box attack')
parser.add_argument('--model-dir', default='./data-model/test', help='model for white-box attack evaluation')
parser.add_argument('--source-model-path', default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path', default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')

args = parser.parse_args()

# settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True}


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size, restarts=1, attack_type=args.attack):
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    out = fmodel(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    #####################################################################################################
    if attack_type == 'pgd':
        attack = fb.attacks.PGD(abs_stepsize=step_size, steps=num_steps, random_start=bool(restarts))
        _, X_adv, _ = attack(fmodel, X, y, epsilons=epsilon)

    elif attack_type == 'pgd_l1':
        attack = fb.attacks.L1PGD(abs_stepsize=step_size, steps=num_steps, random_start=bool(restarts))
        _, X_adv, _ = attack(fmodel, X, y, epsilons=epsilon)
    elif attack_type == 'pgd_l2':
        attack = fb.attacks.L2PGD(abs_stepsize=step_size, steps=num_steps, random_start=bool(restarts))
        _, X_adv, _ = attack(fmodel, X, y, epsilons=epsilon)
    elif attack_type == 'deepfool':
        attack = fb.attacks.LinfDeepFoolAttack()
        _, X_adv, _ = attack(fmodel, X, y, epsilons=epsilon)
    elif attack_type == 'cw_l2':
        attack = fb.attacks.L2CarliniWagnerAttack(steps=num_steps)
        _, X_adv, _ = attack(fmodel, X, y, epsilons=epsilon)
    elif attack_type == 'cw_li':
        std = torch.tensor((1., 1., 1.)).view(3, 1, 1).cuda()
        eps = epsilon / std
        delta = utils.attack_pgd_cw(fmodel, X, y, epsilon=eps, alpha=step_size,
                                    attack_iters=num_steps, restarts=restarts, num_classes=args.num_classes)
        X_adv = X + delta
    elif attack_type == 'fgsm':
        attack = fb.attacks.FGSM()
        _, X_adv, _ = attack(fmodel, X, y, epsilons=epsilon)
    elif attack_type == 'autoattack':
        from autoattack import AutoAttack
        adversary = AutoAttack(fmodel, norm='Linf', eps=epsilon, version='standard')
        X_adv = adversary.run_standard_evaluation(X, y, bs=X.size(0))

    #####################################################################################################

    err_pgd = (fmodel(X_adv).data.max(1)[1] != y.data).float().sum()

    if args.vis:
        print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size, restarts=args.restarts, attack_type=args.attack):
    model_source.eval()
    model_target.eval()

    model_source = fb.PyTorchModel(model_source, bounds=(0, 1))
    model_target = fb.PyTorchModel(model_target, bounds=(0, 1))

    out_source = model_source(X).data.max(1)[1]
    out_target = model_target(X).data.max(1)[1]

    err_source = (out_source != y.data).float().sum()
    err_target = (out_target != y.data).float().sum()

    if attack_type == 'pgd':
        attack = fb.attacks.PGD(abs_stepsize=step_size, steps=num_steps, random_start=bool(restarts))
        _, X_adv, _ = attack(model_source, X, y, epsilons=epsilon)
    elif attack_type == 'deepfool':
        attack = fb.attacks.LinfDeepFoolAttack()
        _, X_adv, _ = attack(model_source, X, y, epsilons=epsilon)
    elif attack_type == 'cw_l2':
        attack = fb.attacks.L2CarliniWagnerAttack(steps=num_steps)
        _, X_adv, _ = attack(model_source, X, y, epsilons=epsilon)
    elif attack_type == 'cw_li':
        std = torch.tensor((1., 1., 1.)).view(3, 1, 1).cuda()
        eps = epsilon / std
        delta = utils.attack_pgd_cw(model_source, X, y, epsilon=eps, alpha=step_size,
                                    attack_iters=num_steps, restarts=restarts, num_classes=args.num_classes)
        X_adv = X + delta
    elif attack_type == 'fgsm':
        attack = fb.attacks.FGSM()
        _, X_adv, _ = attack(model_source, X, y, epsilons=epsilon)

    pgd_pred_source = model_source(X_adv).data.max(1)[1]
    pgd_pred_target = model_target(X_adv).data.max(1)[1]

    err_pgd_source = (pgd_pred_source != y.data).float().sum()
    err_pgd_target = (pgd_pred_target != y.data).float().sum()

    adv_num_source = (pgd_pred_source != out_source).float().sum()
    adv_num_target = (pgd_pred_target != out_target).float().sum()
    return err_source.item(), err_target.item(), err_pgd_source.item(), err_pgd_target.item(), adv_num_source.item(), adv_num_target.item()


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    n = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=False), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        n += y.size(0)

        if args.vis:
            print('natural_acc: {:.4f}, robust_acc: {:.4f}'.format(1 - natural_err_total / n, 1 - robust_err_total / n))

    # print('natural_err_total: ', natural_err_total)
    # print('robust_err_total: ', robust_err_total)

    print('Nat. Acc.: ', 100 - natural_err_total / 100)
    print('Adv. Acc.: ', 100 - robust_err_total / 100)


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()

    natural_err_total_source = 0
    natural_err_total_target = 0
    robust_err_total_source = 0
    robust_err_total_target = 0
    pgd_num_total_source = 0
    pgd_num_total_target = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=False), Variable(target)

        err_natural_source, err_natural_target, err_robust_source, err_robust_target, pgd_num_source, pgd_num_target = _pgd_blackbox(
            model_target,
            model_source, X, y)

        natural_err_total_source += err_natural_source
        natural_err_total_target += err_natural_target

        robust_err_total_source += err_robust_source
        robust_err_total_target += err_robust_target

        pgd_num_total_source += pgd_num_source
        pgd_num_total_target += pgd_num_target

    print(
        "src-nat-err \t tar-nat-err \t src-robust-err \t tar-robust-err \t adv src-pgd-adv \t tar-pgd-adv \t tranfer-ratio")
    print("{:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
        natural_err_total_source, natural_err_total_target,
        robust_err_total_source,
        robust_err_total_target, pgd_num_total_source, pgd_num_total_target,
        pgd_num_total_target / pgd_num_total_source
    ))


train_loader, test_loader = utils.get_loaders(args.data_dir, args.batch_size, args.dataset)

if args.dataset == 'tinyimagenet':
    stride = 2
else:
    stride = 1

if args.white_box_attack:
    print('pgd white-box attack')
    # model = model_factory.create_model(name=args.arch, num_classes=args.num_classes).to(device)
    model = WideResNet(num_classes=args.num_classes, stride=stride).to(device)
    model.load_state_dict(torch.load(args.model_dir))

    eval_adv_test_whitebox(model, device, test_loader)
else:
    print('pgd black-box attack')
    model_source = WideResNet(num_classes=args.num_classes, stride=stride).to(device)
    model_source.load_state_dict(torch.load(args.source_model_path))

    model_target = WideResNet(num_classes=args.num_classes, stride=stride).to(device)
    model_target.load_state_dict(torch.load(args.target_model_path))

    eval_adv_test_blackbox(model_target, model_source, device, test_loader)
