
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import datetime


def g_loss_fn(p):
    return torch.mean(-torch.log(p))

def d_loss_fn(p1, p2):
    return g_loss_fn(p1) - torch.mean(torch.log(1 - p2))

adv_loss = torch.nn.BCELoss()


def training_loop(n_epochs, n_steps, bs, lr, data, discriminator, generator):

    #preperations
    loader = torch.utils.data.DataLoader(data, bs, shuffle=True)
    
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(1, n_epochs+1):
        for i, (real_batch, _) in enumerate(loader):
            bs = real_batch.shape[0]
            for _ in range(1, n_steps+1):
                noise_batch = torch.randn(bs, 100)
                d_loss = d_loss_fn(discriminator(real_batch), discriminator(generator(noise_batch.detach())))
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            noise_batch = torch.rand(bs, 100)
            g_loss = g_loss_fn(discriminator(generator(noise_batch)))
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if i == 0 or (i + 1) % 50 == 0:
                print(f'{datetime.datetime.now()} Epoch: {epoch}|{n_epochs}, Batch: {i}|{len(loader)}, Discriminator Loss: {d_loss.item()} Generator Loss: {g_loss.item()}')

            batches_done = (epoch - 1)* len(loader) + i
            if batches_done % 400 == 0:
                fake_imgs = generator(noise_batch)
                save_image(fake_imgs[:25], f'output/bd_{batches_done}.png', nrow=5, normalize=True )


if __name__ == "__main__":
    
    path = 'mnist'
    mnist_raw = datasets.MNIST(path, train=True, download=False, transform=transforms.ToTensor())
    mnist_raw_te = datasets.MNIST(path, train=False, download=False, transform=transforms.ToTensor())

    # Calculate mean and std for normalization.
    stacked_imgs = torch.stack([img_t for img_t, _ in mnist_raw], 3) # stack images into 4th dim
    mean = stacked_imgs.view(1,-1).mean(dim=1) # view reshapes to a 3 x (collapsed) tensor 
    std = stacked_imgs.view(1,-1).std(dim=1)
    #print(f'mean {mean} std: {std}')

    # Import tensor normalized data
    mnist = datasets.MNIST(path, train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        ]))
    mnist_te = datasets.MNIST(path, train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ]))

    # tensor info
    img, _ = mnist[0]
    [n_channels, height, width] = list(img.size())

    dis_dict = OrderedDict([
        ('flat1', nn.Flatten()),
        ('lin1', nn.Linear(784, 512)),
        ('act1', nn.LeakyReLU(negative_slope=0.2)),
        ('lin2', nn.Linear(512, 256)),
        ('act2', nn.LeakyReLU(negative_slope=0.2)),
        ('lin3', nn.Linear(256, 1)),
        ('act3', nn.Sigmoid())])

    gen_dict = OrderedDict([
        ('lin1', nn.Linear(100, 128)),
        ('act1', nn.LeakyReLU(negative_slope=0.2)),
        ('bnorm1', nn.BatchNorm1d(128, momentum=0.8)),
        ('lin2', nn.Linear(128, 256)),
        ('act2', nn.LeakyReLU(negative_slope=0.2)),
        ('bnorm2', nn.BatchNorm1d(256, momentum=0.8)),
        ('lin3', nn.Linear(256, 512)),
        ('act3', nn.LeakyReLU(negative_slope=0.2)),
        ('bnorm3', nn.BatchNorm1d(512, momentum=0.8)),
        ('lin4', nn.Linear(512, 1024)),
        ('act4', nn.LeakyReLU(negative_slope=0.2)),
        ('bnorm4', nn.BatchNorm1d(1024, momentum=0.8)),
        ('act5', nn.LeakyReLU(negative_slope=0.2)),
        ('lin5', nn.Linear(1024, height*width)),
        ('act6', nn.Tanh()),
        ('unflat1', nn.Unflatten(1, (1, height, width)))])

    discriminator = nn.Sequential(dis_dict)
    generator = nn.Sequential(gen_dict)

    training_loop(n_epochs=10, 
            n_steps=1, 
            bs=64, 
            lr=2e-4, 
            data=mnist, 
            discriminator=discriminator, 
            generator=generator)
    

    noise_batch = torch.randn(25, 100)
    fake_imgs = generator(noise_batch).detach()
    fake_imgs = transforms.Normalize(-1,2)(fake_imgs)
    fake_imgs =  fake_imgs.squeeze(1)
    fake_imgs_plt = fake_imgs.permute(1, 2, 0)

    fig, _ = mnist[0]
    data = torch.stack([img for img, _ in mnist], 0)
    data = data[:25].squeeze(1)
    data_plt = data.permute(1, 2, 0)

    for i in range(25):
        plt.subplot(5, 5, 1+i)
        plt.axis('off')
        # plt.imshow(data_plt[:,:,i], cmap='gray')
        plt.imshow(fake_imgs_plt[:,:,i], cmap='gray')

    plt.show()

