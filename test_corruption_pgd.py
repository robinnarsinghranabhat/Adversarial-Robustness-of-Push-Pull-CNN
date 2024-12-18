import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# from alexnet import alexnet
from resnet.resnetcifar import *
from densenet.densenetcifar import *
# from wideresnet.wideresnet import *
from datasets.cifarcorrupted import CIFAR10_C
from datasets.cifarcorrupted import CIFAR100_C
from datasets.noisycifar import NCIFAR10
from datasets.noisycifar import NCIFAR100


parser = argparse.ArgumentParser(description='Test on CIFAR-10-C')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corrupted-data-dir', default='', type=str, help='root path of the CIFAR-C dataset')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=50, type=int, help='print frequency (default: 10)')

parser.add_argument('--layers', default=20, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor (default: 1)')

parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='To not use bottleneck block')
parser.add_argument('--no-efficient', dest='efficient', action='store_false', help='to not use efficient impl.')

parser.add_argument('--pushpull', action='store_true', help='use Push-Pull as 1st layer (default: False)')
parser.add_argument('--use-cuda', action='store_true', help='Use Cuda (default: False)')
parser.add_argument('--pp-block1', action='store_true', help='use 1st PushPull residual block')
parser.add_argument('--pp-block1-reduced', action='store_true', help='use 1st PushPull residual block reduced')
parser.add_argument('--pp-all', action='store_true', help='use all PushPull residual block')
parser.add_argument('--pp-all-reduced', action='store_true', help='use all PushPull residual block reduced')
parser.add_argument('--modelfile', default='checkpoint', type=str, help='name of the file of the model')
parser.add_argument('--alpha-pp', default=1, type=float, help='inhibition factor (default: 1.0)')
parser.add_argument('--pgd-epsilon', default=0.0, type=float, help='Epsilon for PGD attack. (default: 0.0)') 

parser.add_argument('--scale-pp', default=2, type=float, help='upsampling factor for PP kernels (default: 2)')

parser.add_argument('--train-alpha', action='store_true', help='train alpha of push-pull kernels (Default: False)')

parser.add_argument('--lpf-size', default=None, type=int, help='Size of the LPF for anti-aliasing (default: 1)')

parser.add_argument('--arch', default='resnet', type=str, help='architecture (resnet, densenet, ...)')
parser.add_argument('--name', default='01-20', type=str, help='name of experiment-model')

best_prec1 = 0
args = parser.parse_args()

pgd_epsilon = args.pgd_epsilon  # Maximum allowed perturbation    
use_cuda = torch.cuda.is_available() & args.use_cuda

if use_cuda:
    print("Using CUDA Environment")

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    # 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    # 'snow', 'frost', 'fog', 'brightness',
    # 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]
# Root folder of the CIFAR-C and CIFAR-P data sets
# Please change it with the path to the folder where you un-tar the CIFAR-C data set
# (or use the --corrupted-data-dir argument)
corr_dataset_root = ''
if args.corrupted_data_dir != '':
    corr_dataset_root = args.corrupted_data_dir

pgd_alpha = 0.01   # Step size
pgd_iters = 10     # Number of iterations


def main():
    # Clean Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
    if args.dataset == 'cifar10':
        clean_data = NCIFAR10('./data', train=False, transform=transform_test, normalize_transform=normalize)
        nclasses = 10

    elif args.dataset == 'cifar100':
        clean_data = NCIFAR100('./data', train=False, transform=transform_test, normalize_transform=normalize)
        nclasses = 100

    clean_loader = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    # --------------------------------------------------------------------------------
    # create model
    expdir = ''
    if args.arch == 'resnet':
        expdir = 'experiments/resnet-cifar/'
        rnargs = {'use_pp1': args.pushpull,
                  'pp_block1': args.pp_block1,
                  'pp_all': args.pp_all,
                  'train_alpha': args.train_alpha,
                  'size_lpf': args.lpf_size}

        if args.layers == 20:
            model = resnet20(**rnargs)
        elif args.layers == 32:
            model = resnet32(**rnargs)
        elif args.layers == 44:
            model = resnet44(**rnargs)
        elif args.layers == 56:
            model = resnet56(**rnargs)
    elif args.arch == 'densenet':
        expdir = 'models/densenet-cifar/'
        rnargs = {'use_pp1': args.pushpull,
                  'pp_block1': args.pp_block1,
                  'num_classes': nclasses,
                  'small_inputs': True,
                  'efficient': args.efficient,
                  'compression': args.reduce,
                  'drop_rate': args.droprate,
                  'scale_pp': args.scale_pp,
                  'alpha_pp': args.alpha_pp
                  }

        if args.layers == 40:
            model = densenet40_12(**rnargs)
        elif args.layers == 100:
            if args.growth == 12:
                model = densenet100_12(**rnargs)
            elif args.growth == 24:
                model = densenet100_24(**rnargs)
    elif args.arch == 'alexnet':
        expdir = 'models/alexnet-cifar/'
        # model = alexnet.AlexNet(num_classes=nclasses)
    else:
        raise RuntimeError('Fatal error - no other networks implemented')

    # load trained parameters in the model
    if use_cuda:
        trained_model = torch.load(expdir + '%s/' % args.name + args.modelfile + '.pth.tar')
    else:
        trained_model = torch.load(expdir + '%s/' % args.name + args.modelfile + '.pth.tar',
                                   map_location=lambda storage, loc: storage)

    # ------------------ Start loading model ---------------
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    trained_model['state_dict'] = {k: v for k, v in trained_model['state_dict'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(trained_model['state_dict'])
    model.load_state_dict(trained_model['state_dict'])
    # ------------------ Finish loading model --------------

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()  # define loss function (criterion) and optimizer

    # evaluate on validation set
    fileout = open(expdir + args.name + '/test_clean.txt', "a+")
    prec1 = validate(clean_loader, model, criterion, adversarial_eps=pgd_epsilon, file=fileout)
    print('Overall Test accuracy on clean CIFAR : \n{}'.format(prec1))
    # fileout.write('Test accuracy clean: \n{}'.format(prec1))
    # fileout.close()

    # ------------------------------------------------------------------
    #           VALIDATE ON CIFAR-10-C
    # ------------------------------------------------------------------
    # error_rates = []
    mean_rates = []

    # accuracy_rates = []
    mean_acc_rates = []

    f1 = open(expdir + args.name + '/C_details.txt', "w+")
    f2 = open(expdir + args.name + '/C_average.txt', "w+")
    if pgd_epsilon == 0:
        print("Evaluating on NOISY CIFAR Images")
    else:
        print(f"Evaluating on NOISY CIFAR Images perturbed by PGD at epsilon : {pgd_epsilon}")

    for distortion_name in distortions:
        rates, accuracies = validate_corrupted(distortion_name, model, criterion, adversarial_eps=pgd_epsilon)
        # error_rates.append(rates)
        mean_rates.append(np.mean(rates))

        # accuracy_rates.append(accuracies)
        mean_acc_rates.append(np.mean(accuracies))

        f1.write(distortion_name + ': ' + ' '.join(map(str, rates)) + '\n')
        f2.write('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}\n'.format(distortion_name, 100 * np.mean(rates)))
        
        print(distortion_name + ' error at severities (1-5): ' + ' '.join(['%.4f ' % s for s in rates]))
        print('Error-Rates for Distortion: {:15s} : {:.2f}'.format(distortion_name, 100 * np.mean(rates)))

        print(distortion_name + ' accuracy at severities (1-5): ' + ' '.join(['%.4f ' % s for s in accuracies]))
        print('Mean Accuracy for Distortion: {:15s} across all severities : {:.2f}'.format(distortion_name, np.mean(accuracies)))


    f1.close()
    f2.close()
    print('Overall Mean Error on all distorted CIFAR images at each severity  (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(mean_rates)))
    print('Overall Mean Accuracty on all distorted CIFAR images at each severity: {:.2f}'.format(np.mean(mean_acc_rates)))


def validate_corrupted(distortion_name, model, criterion=None, adversarial_eps=0, alpha=pgd_alpha):
    print("-- Testing Acc. on ", distortion_name, ' --')
    ## These are average erros and acc. for each severity for particular distortion.
    errs = []
    accuracies = []

    # Data loading code
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
    ])
    if args.dataset == 'cifar10':
        dataset = CIFAR10_C(corr_dataset_root, transform=transform_test, corr_category=distortion_name)
    elif args.dataset == 'cifar100':
        dataset = CIFAR100_C(corr_dataset_root, transform=transform_test, corr_category=distortion_name)

    for severity in range(1, 6):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        print("at severity : ", severity)
        distorted_dataset = dataset.get_severity_set(severity)

        kwargs = {'num_workers': 0, 'pin_memory': True}
        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            if use_cuda:
                    data = data.cuda()
            
            model.eval()
            if adversarial_eps == 0:
                with torch.no_grad():
                    output = model(data)
                loss = criterion(output, target)
            

            else:  # Add Adversarial perturbation
                original_images = data.clone().detach()
                perturbed_images = data.clone().detach()
                if use_cuda:
                    original_images = original_images.cuda()
                    perturbed_images = perturbed_images.cuda()
                perturbed_images.requires_grad = True

                for _ in range(pgd_iters):
                    output = model(perturbed_images)
                    loss = criterion(output, target)
                    
                    model.zero_grad()
                    loss.backward()
                    grad = perturbed_images.grad.data

                    # Update the perturbed images
                    perturbed_images = perturbed_images + alpha * grad.sign()
                    
                    # Project back to the l_infinity ball
                    perturbation = torch.clamp(perturbed_images - original_images, -adversarial_eps, adversarial_eps)
                    perturbed_images = torch.clamp(original_images + perturbation, 0, 1).detach()
                    perturbed_images.requires_grad = True

            
            # measure accuracy
            prec1 = accuracy(output, target, topk=(1,))[0]
            top1.update(prec1.item(), data.size(0))

            pred = output.detach().cpu().max(1)[1]
            correct += pred.eq(target.cpu()).sum()

            if batch_idx % args.print_freq == 0:
                print("INSIDE LOOP, Acc > ", top1.avg)
        
        
        accuracies.append(top1.avg)
        errs.append(1 - correct.numpy() / len(distorted_dataset))
    return errs, accuracies


def validate(val_loader, model, criterion, adversarial_eps=0, file=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if adversarial_eps == 0:
        print("Evaluating on CLEAN CIFAR Images")
    else:
        print(f"Evaluating on CLEAN CIFAR Images perturbed by PGD at epsilon : {adversarial_eps}")

    # This makes dropout and batch normalization constant. 
    # Thus leads to stable, reproducible gradients for inputs.
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda() # Temporarilly removed : async=True
            input = input.cuda()
        
        input.requires_grad = True

        # compute output
        if adversarial_eps == 0:
            with torch.no_grad():
                output = model(input)
            loss = criterion(output, target)
        else:  # Add Adversarial perturbation
    
            original_images = input.clone().detach()
            perturbed_images = input.clone().detach()
            if use_cuda:
                original_images = original_images.cuda()
                perturbed_images = perturbed_images.cuda()

            perturbed_images.requires_grad = True

            for _ in range(pgd_iters):
                output = model(perturbed_images)
                loss = criterion(output, target)
                
                model.zero_grad()
                loss.backward()
                grad = perturbed_images.grad.data

                # Update the perturbed images
                perturbed_images = perturbed_images + pgd_alpha * grad.sign()
                
                # Project back to the l_infinity ball
                perturbation = torch.clamp(perturbed_images - original_images, -adversarial_eps, adversarial_eps)
                perturbed_images = torch.clamp(original_images + perturbation, 0, 1).detach()
                perturbed_images.requires_grad = True
                

        loss = loss.detach()
        output = output.detach()

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader),
                                                                  batch_time=batch_time, loss=losses,
                                                                  top1=top1))
            if file is not None:
                file.write('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) \n'.format(i, len(val_loader),
                                                                              batch_time=batch_time, loss=losses,
                                                                              top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    file.write(' * Prec@1 {top1.avg:.3f} \n'.format(top1=top1))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
