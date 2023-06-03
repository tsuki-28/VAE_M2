import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from gan_classification.gan_utils import get_mnist
from _collections import defaultdict
import matplotlib.pyplot as plt #绘图
import pandas as pd
from tensorboardX import SummaryWriter
# os.makedirs("images", exist_ok=True)

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# opt = parser.parse_args()
# print(opt)

# cuda = True if torch.cuda.is_available() else False

n_epochs = 200
lr = 0.001
b1 = 0.5
b2 = 0.999
latent_dim = 100
n_classes = 10
imag_size = 1000
sample_interval = 10

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        # self.init_size = img_size // 4  # Initial size before upsampling
        # self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(

            # nn.BatchNorm2d(128),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(64, channels, 3, stride=1, padding=1),
            # nn.Tanh(),
            nn.Linear(100, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 1000),
            # nn.BatchNorm1d(1000),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        # out = self.l1(gen_input)
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        img = self.conv_blocks(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # def discriminator_block(in_filters, out_filters, bn=True):
        #     """Returns layers of each discriminator block"""
        #     block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
        #     if bn:
        #         block.append(nn.BatchNorm2d(out_filters, 0.8))
        #     return block

        self.conv_blocks = nn.Sequential(
            # *discriminator_block(channels, 16, bn=False),
            # *discriminator_block(16, 32),
            # *discriminator_block(32, 64),
            # *discriminator_block(64, 128),
            nn.GRU(input_size=2, hidden_size=100, num_layers=3, batch_first=True, dropout=0.5),
            # nn.Linear(100, 10)

        )

        # The height and width of downsampled image
        # ds_size = img_size // 2 ** 4

        # Output layers
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        # self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.adv_layer = nn.Sequential(nn.Linear(100,1),nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(100, n_classes),nn.Softmax())

    def forward(self, img):
        img=img.view(img.size(0),2,-1)
        img=img.permute(0,2,1)
        _,out = self.conv_blocks(img)
        # out = out.view(out.shape[0], -1)
        out=out[2]
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# # Loss functions
# adversarial_loss = torch.nn.BCELoss()
# auxiliary_loss = torch.nn.CrossEntropyLoss()
#
# # Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#     auxiliary_loss.cuda()
#
# # Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)
#
# # Configure data loader
# # os.makedirs("../../data/mnist", exist_ok=True)
# # dataloader = torch.utils.data.DataLoader(
# #     datasets.MNIST(
# #         "ministdata",
# #         train=True,
# #         download=True,
# #         transform=transforms.Compose(
# #             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
# #         ),
# #     ),
# #     batch_size=opt.batch_size,
# #     shuffle=True,
# # )
# labelled,test_set = get_mnist(batch_size=100,labels_per_class=10)
#
# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#
# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def test_M2(model, dataloader,  cuda_available: bool,test_data: defaultdict,  verbose=True):
    import torch.nn.functional as F
    # init
    running_loss = 0
    correct = 0
    total = 0
    loss_class = 0
    saved = False
    with torch.no_grad():
        model.eval()
        for i, (input, target) in enumerate(dataloader):
            if cuda_available:
                model = model.cuda()
                input = input.cuda()
                target = target.cuda()
            input=torch.Tensor.float(input)
            # loss_lab, _, _, _ = model(input, target)
            # loss_unlab, _, _, _, = model(input)

            _,output = model(input)
            classifier_loss = F.cross_entropy(output, target)

            loss_class += classifier_loss.item()
            # loss = loss_lab + alpha * classifier_loss + loss_unlab
            loss =  classifier_loss
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            running_loss += loss.item()

        try:
            max_acc = max(test_data["test_accuracy"])
        except:
            max_acc = 0
        i=i+1
        test_data["Tot_loss"] += [100*running_loss /(i+1) ]
        test_data["classifier_loss"] += [100*loss_class /(i+1) ]
        test_data["test_accuracy"] += [100 * correct.true_divide(total).item()]

        current_acc = test_data["test_accuracy"][-1]
        if current_acc >= max_acc:
            torch.save(model.state_dict(), "./acgan_classifier.pt")
            max_acc = current_acc
            saved = True

    if verbose:
        print("Test :")
        print(
            "Tot loss : {}, Classifier loss : {}, Classifier accuracy : {}".format(round(test_data["Tot_loss"][-1], 3),
                                                                                   round(
                                                                                       test_data["classifier_loss"][-1],
                                                                                       3),
                                                                                   round(test_data["test_accuracy"][-1],
                                                                                         2)))
        if saved:
            print(f"Saved Checkpoint with accuracy {max_acc}")


# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     # Sample noise
#     z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim)))
#     # Get labels ranging from 0 to n_classes for n rows
#     labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#     labels =LongTensor(labels)
#     gen_imgs = generator(z, labels)
#     save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
def gen_img_plot(model,test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())

    # fig = plt.figure(figsize=(4,4))
    #
    # for i in range(16):
    #         plt.subplot(4,4,i+1)
    #         plt.imshow((prediction[i]+1)/2)
    #         plt.axis('off')
    prediction = np.squeeze(model(test_input).detach().cpu())
    prediction=prediction.view(prediction.size(0), 2, 500)
    prediction=prediction.numpy()
    fig,ax=plt.subplots(4,4,figsize=(8,8))
    for i,ax in enumerate(ax.flat):
        ax.plot(prediction[i,0])
        ax.plot(prediction[i,1])
        path = "../gan_classification/gener"
        name = str(i) + '.csv'
        path=os.path.join(path,name)
        pd.DataFrame(prediction[i]).round(5).to_csv(path, header=False, index=False)
    plt.show()

if __name__ == '__main__':

    cuda = True if torch.cuda.is_available() else False
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    # os.makedirs("../../data/mnist", exist_ok=True)
    # dataloader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "ministdata",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    #         ),
    #     ),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    # )
    labelled, test_set = get_mnist(batch_size=100, labels_per_class=10)
    test_data = defaultdict(list)
    # Optimizers
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    writer = SummaryWriter('acgan')



    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(labelled):

            batch_size = imgs.size(0)

            # Adversarial ground truths
            # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = imgs.type(FloatTensor)
            labels = labels.type(LongTensor)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            # size = imgs.size(0)  # img的第一位是size,获取批次的大小
            z= FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))
            # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            # gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
            gen_labels = LongTensor(np.random.randint(0, n_classes, batch_size))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            # gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity,
                                  torch.ones_like(validity)) + auxiliary_loss(pred_label, gen_labels))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, torch.ones_like(real_pred)) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, torch.zeros_like(fake_pred)) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            b=np.max(real_aux.data.cpu().numpy(),axis=1)
            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc_withgen = np.mean(np.argmax(pred, axis=1) == gt)
            d_acc=np.mean(np.argmax(real_aux.data.cpu().numpy(),axis=1)==labels.data.cpu().numpy())

            d_loss.backward()
            optimizer_D.step()

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
            # )
            # batches_done = epoch * len(labelled) + i
            # if batches_done % sample_interval == 0:
            #     sample_image(n_row=10, batches_done=batches_done)
        print(
            "[Epoch %d/%d] [D loss: %f, acc: %d%%, acc_withgen: %d%%] [G loss: %f]"
            % (epoch, n_epochs,d_loss.item(), 100 * d_acc, 100*d_acc_withgen, g_loss.item())
        )
        with torch.no_grad():
            writer.add_scalars("dis_loss", {"train": np.round(d_loss.numpy(), 3)}, epoch+1)
            writer.add_scalars("gen_loss", {"train": np.round(g_loss.numpy(), 3)}, epoch+1)
        test_M2(discriminator,test_set,cuda,test_data)


