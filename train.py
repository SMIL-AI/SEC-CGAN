import argparse
import os
from multiprocessing import Process
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import *
from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimension of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--ngf", type=int, default=64, help="size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="size of feature maps in discriminator")
    parser.add_argument("--multiplier", type=float, default=0.6, help="weighting multiplier, which controls the relative contribution of generated data to the classifier training ")
    parser.add_argument("--threshold", type=float, default=0.7, help="confidence threshold, which controls the quality of data to be used for classifier training")
    parser.add_argument("--datasize", type=float, default=0.1, help="datasize")
    opt, unknown = parser.parse_known_args()
    return opt

opt = get_args()
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2], opt.n_classes)

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)
netC = ResNet18()

generator.to(device)
discriminator.to(device)
netC.to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optC = torch.optim.Adam(netC.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay = 1e-3)

# Loss functions
adversarial_loss = torch.nn.BCELoss()
criterion = nn.CrossEntropyLoss()

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

# define data augmentation and normalization
transform = transforms.Compose([
        transforms.Resize(32),                     # Resize to the same size
        transforms.RandomCrop(32, padding=4),      # Crop to get square area
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])


# regular data loaders
batch_size = opt.batch_size
flag = True
if os.path.exists("datasets/SVHN"): flag = False
trainset = datasets.SVHN("datasets/SVHN", split='train', download = flag, transform=transform)
traindataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.SVHN("datasets/SVHN", split='test', download = flag, transform=transform_test)
testloader= torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)

#create subsets
dataSizeConstant = opt.datasize
subTrainSet,_ = torch.utils.data.random_split(trainset, [int(dataSizeConstant*len(trainset)), len(trainset)-int(dataSizeConstant*len(trainset))])

# classes-balanced sampling
def cb_sampling():
    imgs = []
    for data in subTrainSet:
            img, target = data
            imgs.append((img, target))
    weights = make_weights_for_balanced_classes(imgs, 10)
    weights = torch.DoubleTensor(weights)

    subTrainLoader = gain_sample_w(subTrainSet, batch_size= batch_size, weights=weights)
    return subTrainLoader

# validation routine
def validate():
    netC.eval()
    correct = 0
    total = 0
    global gpred_labels, greal_labels
    gpred_labels = torch.empty(0).to(device)
    greal_labels = torch.empty(0).to(device)
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = netC(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            gpred_labels = torch.cat((gpred_labels, torch.flatten(predicted)))
            greal_labels = torch.cat((greal_labels, torch.flatten(labels)))
    print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))

# ------------------------------
#  Training
# ------------------------------
G_losses = []
D_losses = []
C_losses = []

loader = cb_sampling()

def main():
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(loader):

            batch_size = imgs.shape[0]
            validlabel = Variable(FloatTensor(batch_size, ).fill_(1.0), requires_grad=False).to(device)
            fakelabel = Variable(FloatTensor(batch_size, ).fill_(0.0), requires_grad=False).to(device)

            real_imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = Variable(labels.type(LongTensor)).to(device)

            # ---------------------------------
            # (1) Update D network:
            # ---------------------------------
            # Train with all-real data batch
            discriminator.zero_grad()
            # Forward pass real batch through D
            output = discriminator(real_imgs, labels, opt).view(-1)
            # Calculate loss on all-real batch
            errD_real = adversarial_loss(output, validlabel)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            # Train with all-fake data batch
            # Generate batch of latent vectors and fake labels
            z = torch.randn(batch_size, opt.latent_dim, 1, 1, device=device)
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
            # Generate fake image batch with G
            fake = generator(z, labels)
            # Discriminate all fake batch with D
            output = discriminator(fake.detach(), labels, opt).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = adversarial_loss(output, fakelabel)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward(retain_graph=True)
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_D.step()

            # ---------------------------------
            # (2) Update G network:
            # ---------------------------------
            generator.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake, labels, opt).view(-1)
            # Calculate G's loss based on this output
            errG = adversarial_loss(output, validlabel)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizer_G.step()

            # ---------------------------------
            # (3) Updata C network:
            # ---------------------------------
            # train classifier on real data
            fake = fake.detach().clone()
            predictions = netC(real_imgs)
            realClassifierLoss = criterion(predictions, labels)
            realClassifierLoss.backward(retain_graph=True)

            optC.step()
            optC.zero_grad()

            # train classifier on the synthesized data selected by the discriminator with a confidence being greater than or equal to β.
            x = output.ge(opt.threshold)
            Drealfake = fake[x]
            Dreallabels = labels[x]
            if Drealfake.shape[0] != 0:
                predictionsFake = netC(Drealfake)
                fakeClassifierLoss = criterion(predictionsFake, Dreallabels) * opt.multiplier
                fakeClassifierLoss.backward()
                optC.step()
                optC.zero_grad()
            if i % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f]"
                    % (epoch, opt.n_epochs, i, len(loader), errD.item(), errG.item(), realClassifierLoss.item()))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            C_losses.append(realClassifierLoss.item())
            batches_done = epoch * len(loader) + i
        validate()

def plot():
    # Plot the training progression
    plt.style.use('classic')
    plt.rcParams['figure.facecolor'] = 'white'

    plt.figure(figsize=(20,5))
    plt.subplot(1, 2, 1)

    plt.title("Generator and Discriminator Loss During Training")
    line1 =plt.plot(G_losses,'b-')
    line2 =plt.plot(D_losses,'r-')
    plt.legend(labels=['Generator_loss', ' Discriminator_loss'])
    plt.xlabel("iterations")

    plt.subplot(1, 2, 2)
    line3 =plt.plot(C_losses,'g-')
    plt.title('Classifier loss')
    plt.legend(labels=['Classifier_loss'])
    plt.xlabel("iterations")

    plt.savefig('results/images/Conditional_classifier_GAN.png',dpi=400,bbox_inches='tight')
    plt.show()

def save():
    # save the trained models
    torch.save(generator, f'results/models/generator_{opt.datasize}.pth')
    torch.save(discriminator, f'results/models/discriminator_{opt.datasize}.pth')
    torch.save(netC, f'results/models/netC_{opt.datasize}.pth')

if __name__ == '__main__':
    # create and configure a new process
    process = Process(target=cb_sampling)
    # start the new process
    process.start()
    # wait for the new process to finish
    process.join()
    main()
    plot()
    save()