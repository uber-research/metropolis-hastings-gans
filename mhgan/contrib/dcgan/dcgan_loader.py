# Modifications Copyright (c) 2018 Uber Technologies, Inc.
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from dcgan import gan_trainer

SEED_MAX = 2**32 - 1


def get_opts(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')  # noqa: E402
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')  # noqa: E402
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')  # noqa: E402
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')  # noqa: E402
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')  # noqa: E402
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')  # noqa: E402
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')  # noqa: E402
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')  # noqa: E402
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')  # noqa: E402
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')  # noqa: E402
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')  # noqa: E402
    parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')  # noqa: E402
    parser.add_argument('--manualSeed', required=True, type=int, help='manual seed')  # noqa: E402

    opt = parser.parse_args(args=args)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    return opt


def get_data_loader(dataset, dataroot, workers, image_size, batch_size):
    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset == 'lsun':
        dataset = dset.LSUN(root=dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                            ]))
    elif dataset == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
    elif dataset == 'mnist':
        dataset = dset.MNIST(root=dataroot, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5)),
                             ]))
    elif dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())
    else:
        assert False
    assert dataset

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=int(workers))
    return data_loader


def main():
    opt = get_opts()

    # Set all random seeds: avoid correlated streams ==> must use diff seeds.
    # Note: we are not setting np seed since this appears not to use numpy,
    # but that could create reprod issues if there is latent np.random use.
    random.seed(opt.manualSeed)
    torch.manual_seed(random.randint(0, SEED_MAX))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random.randint(0, SEED_MAX))

    # This is faster but worse for reprod.
    cudnn.benchmark = True

    data_loader = get_data_loader(opt.dataset, opt.dataroot, opt.workers,
                                  opt.imageSize, opt.batchSize)

    device = torch.device('cuda:0' if opt.cuda else 'cpu')
    T = gan_trainer(device=device, data_loader=data_loader,
                    batch_size=opt.batchSize, nz=opt.nz, ngf=opt.ngf,
                    ndf=opt.ndf, lr=opt.lr, beta1=opt.beta1, ngpu=opt.ngpu,
                    netG_file=opt.netG, netD_file=opt.netD, outf=opt.outf)

    for net_G, net_D in T:
        print('epoch done.')


if __name__ == '__main__':
    main()
