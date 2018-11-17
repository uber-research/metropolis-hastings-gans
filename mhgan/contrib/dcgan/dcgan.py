# Modifications Copyright (c) 2018 Uber Technologies, Inc.
from itertools import count
# We could use sum([...], []) instead of concat to avoid np import here
from numpy import concatenate as concat
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

LOG_BLOCK_SIZE = 100
BASE_D = 'base'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def gan_trainer(device, data_loader, batch_size, nz, ngf, ndf, lr, beta1, ngpu,
                netG_file='', netD_file='', outf=None, calib_frac=0.0):
    nc = 3
    criterion = nn.BCELoss()
    # Basically an enum here
    fake_label = 0
    real_label = 1

    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.apply(weights_init)
    if netG_file != '':
        netG.load_state_dict(torch.load(netG_file))

    netD = Discriminator(ngpu, ndf, nc).to(device)
    netD.apply(weights_init)
    if netD_file != '':
        netD.load_state_dict(torch.load(netD_file))

    netD_side = Discriminator(ngpu, ndf, nc).to(device)
    netD_side.apply(weights_init)

    batch_size_fixed = batch_size
    fixed_noise = torch.randn(batch_size_fixed, nz, 1, 1, device=device)

    # Prob don't need to use batch_size_fixed with early binding for default
    # args but doing it this way to be safe.
    def gen_f(batch_size_fixed_=batch_size_fixed):
        noise = torch.randn(batch_size_fixed_, nz, 1, 1, device=device)
        x = netG(noise).detach()
        x = x.cpu().numpy()
        return x

    def gen_disc_f(batch_size_fixed_=batch_size_fixed):
        noise = torch.randn(batch_size_fixed_, nz, 1, 1, device=device)
        x = netG(noise).detach()
        scores = {BASE_D: netD(x).detach().cpu().numpy()}
        x = x.cpu().numpy()
        return x, scores

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD_side = \
        optim.Adam(netD_side.parameters(), lr=lr, betas=(beta1, 0.999))

    # Define chunk used for training vs held out
    training_start = int(calib_frac * len(data_loader))

    # Build calib/test data part of real data
    real_calib_data = []
    for i, data in enumerate(data_loader, 0):
        if i < training_start:
            real_calib_data.append(data)
        else:
            break

    for epoch in count():
        scores_real = {}
        scores_real[BASE_D] = \
            concat([netD(data[0].to(device)).detach().cpu().numpy()
                    for data in real_calib_data])

        # Yield before the training loop, so out first round gives us a glance
        # at the initialization (either random or from checkpoint)
        yield gen_f, gen_disc_f, scores_real

        for i, data in enumerate(data_loader, 0):
            if i < training_start:
                continue

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            ############################
            # (1) Update D side network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD_side.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD_side(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            _ = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD_side(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            _ = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD_side.step()

            print('[%d][%d/%d] Loss_D: %.4f '
                  'Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, i, len(data_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if outf is not None and i % LOG_BLOCK_SIZE == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % outf,
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' %
                                  (outf, epoch),
                                  normalize=True)

        if outf is not None:
            torch.save(netG.state_dict(),
                       '%s/netG_epoch_%d.pth' % (outf, epoch))
            torch.save(netD.state_dict(),
                       '%s/netD_epoch_%d.pth' % (outf, epoch))
