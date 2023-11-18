import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from torch.optim.lr_scheduler import StepLR

from transformer_cwgan.transcwgan import *
from transformer_cwgan.utils import args


device = torch.device('cuda:0')

generator = Generator(device=device, n_class=args.n_class, output_dim=args.data_dim)
generator.to(device)

discriminator = Discriminator(dim=args.data_dim, n_class=args.n_class)
discriminator.to(device)

optim_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen,
                       betas=(args.beta1, args.beta2))

optim_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis,
                       betas=(args.beta1, args.beta2))

lr_scheduler_g = StepLR(optim_gen, 1000, gamma=args.gamma_g, last_epoch=-1)
lr_scheduler_d = StepLR(optim_gen, 500, gamma=args.gamma_d, last_epoch=-1)


def compute_gradient_penalty(D, real_samples, real_labels, fake_samples, phi=1):
    alpha = torch.Tensor(np.random.random(size=(real_samples.size(0), 1))).to(real_samples.get_device())

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, real_labels)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
                                    create_graph=True, retain_graph=True)[0]
    # gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train_gan():
    print('Train GAN...')
    global_steps = 0
    ite = 0
    train_x, train_y = load_data(args.select_num)

    train_set = MyDataset(train_x, train_y)
    # for e in range(args.max_epoch):
    while ite < args.max_iteration_num:
        gen_steps = 0
        generator.train()
        discriminator.train()
        # dataloader
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        t_d_loss = []
        t_g_loss = []
        for index, (sample, y) in enumerate(train_loader):
            real_samples = sample.type(torch.FloatTensor).to(device)
            real_labels = y.type(torch.FloatTensor).unsqueeze(1).to(device)
            real_labels = torch.zeros([len(y), args.n_class]).to(device).scatter_(1, real_labels.type(torch.int64), 1)
            real_labels = y.type(torch.FloatTensor).to(device)
            noise = torch.FloatTensor(np.random.normal(0, 1, size=(real_labels.shape[0], 1, 16, 16))).to(device)
            fake_samples = generator(noise, real_labels.unsqueeze(1))
            real_valid = discriminator(real_samples, real_labels.unsqueeze(1))
            fake_valid = discriminator(fake_samples, real_labels.unsqueeze(1))
            optim_dis.zero_grad()

            # 计算GP
            gp = compute_gradient_penalty(discriminator, real_samples, real_labels, fake_samples, args.phi)
            # W Loss
            loss_dis = torch.mean(fake_valid) - torch.mean(real_valid) + gp * 10 / (args.phi ** 2)
            # 更新D
            loss_dis.backward()
            optim_dis.step()
            lr_scheduler_d.step()
            t_d_loss.append(loss_dis.cpu().item())

            # 更新G
            if global_steps % args.n_critic == 0:
                optim_gen.zero_grad()
                # noise = torch.FloatTensor(np.random.normal(0, 1, [len(y), 1, 16, 16])).to(device)
                fake_samples = generator(noise, real_labels.unsqueeze(1))
                fake_valid = discriminator(fake_samples, real_labels.unsqueeze(1))
                loss_gen = -torch.mean(fake_valid).to(device)
                loss_gen.backward()
                optim_gen.step()
                lr_scheduler_g.step()
                gen_steps += 1
            ite += 1

    torch.save(generator, f'./model/{args.name}_gen_base_{args.select_num}.pth')
    torch.save(discriminator, f'./model/{args.name}_dis_base_{args.select_num}.pth')
