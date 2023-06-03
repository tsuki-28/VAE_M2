import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #优化
import numpy as np
import matplotlib.pyplot as plt #绘图
import torchvision #加载图片
from torchvision import transforms #图片变换
from gan_classification.gan_utils import get_mnist
# from vae_classification.vae_m2 import VAE,M2
from gan_classification.gan_utils import init_weights,training_M2,test_M2,training_VAE,training
from tensorboardX import SummaryWriter

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
        nn.Linear(100,256),
        nn.ReLU(),
        # nn.BatchNorm1d(256),
        nn.Linear(256,512),
        nn.ReLU(),
        # nn.BatchNorm1d(512),
        nn.Linear(512,1000),
        # nn.BatchNorm1d(1000),
        nn.Sigmoid()
        # nn.Tanh()
        # nn.Tanh()#对于生成器，最后一个激活函数是tanh,值域：-1到1
          # 对于生成器，最后一个激活函数是tanh,值域：-1到1
        )
    #定义前向传播
    def forward(self,x):  #x表示长度为100的noise输入
        img = self.main(x)
        img=img.view(-1,1000)#转换成图片的形式
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
        nn.Linear(1000,512),
        nn.ReLU(),
        # nn.BatchNorm1d(512),
        nn.Linear(512,256),
        nn.ReLU(),
        # nn.BatchNorm1d(256),
        nn.Linear(256,1),
        nn.Sigmoid()
        )
    def forward(self,x):
        x =x.view(-1,1000) #展平
        x =self.main(x)
        return x


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


# class Gan(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.gen=Generator()
#         self.dis = Discriminator()
#     def forward(self, x):
#         size = x.size(0)  # img的第一位是size,获取批次的大小
#         random_noise = torch.randn(size, 100, device=device)
#         # loss_fn = torch.nn.BCELoss()
#
#         # 判别器训练(真实图片的损失和生成图片的损失),损失的构建和优化
#         # d_optim.zero_grad()  # 梯度归零
#         # 判别器对于真实图片产生的损失
#         real_output = self.dis(x)  # 判别器输入真实的图片，real_output对真实图片的预测结果
#         # d_real_loss = loss_fn(real_output,
#         #                       torch.ones_like(real_output)
#         #                       )
#         # d_real_loss.backward()  # 计算梯度
#
#         # 在生成器上去计算生成器的损失，优化目标是判别器上的参数
#         gen_img = self.gen(random_noise)  # 得到生成的图片
#         # 因为优化目标是判别器，所以对生成器上的优化目标进行截断
#         fake_output_d = self.dis(gen_img.detach())  # 判别器输入生成的图片，fake_output对生成图片的预测;detach会截断梯度，梯度就不会再传递到gen模型中了
#         # 判别器在生成图像上产生的损失
#         # d_fake_loss = loss_fn(fake_output,
#         #                       torch.zeros_like(fake_output)
#         #                       )
#         # d_fake_loss.backward()
#         # 判别器损失
#         # d_loss = d_real_loss + d_fake_loss
#         # 判别器优化
#         # d_optim.step()
#
#         # 生成器上损失的构建和优化
#         # g_optim.zero_grad()  # 先将生成器上的梯度置零
#         fake_output = self.dis(gen_img)
#         # g_loss = loss_fn(fake_output,
#         #                  torch.ones_like(fake_output)
#         #                  )  # 生成器损失
#         # g_loss.backward()
#         # g_optim.step()
#         # 累计每一个批次的loss
#         # with torch.no_grad():
#         #     d_epoch_loss += d_loss
#         #     g_epoch_loss += g_loss
#         return gen_img,real_output,fake_output_d,fake_output








labelled,test_set = get_mnist(batch_size=100,labels_per_class=10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
# beta=1
# alpha=50
# num_classes=10
# num_features = 1000 #输出层维度
# # num_features = 2  # 输入层通道
# latent_f =80#隐变量的空间维度 重点
# layers_enc = [num_features,512,256]  ##Adding extra layer
# layers_dec = [latent_f+num_classes,256,512]
# layers_dec_vae = [latent_f, 256,512]
# # layer_class=[num_features,300,120]
# vae=VAE(layers_enc,layers_dec_vae,latent_f,num_features,beta)
#
#
#
# init_weights(vae)  # 神经网络需要初始化权重
# print(vae)
# lr = 3e-4
epoch=0
# epoch_vae = 0
# EPOCH_vae = 200
EPOCH=200
lr=0.0005
# optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr)
#
# cuda = torch.cuda.is_available()


if __name__ == '__main__':

    # # 对数据做归一化（-1，1）
    # transform = transforms.Compose([
    #     # 将shanpe为（H,W，C）的数组或img转为shape为（C,H,W）的tensor
    #     transforms.ToTensor(),  # 转为张量并归一化到【0，1】；数据只是范围变了，并没有改变分布
    #     transforms.Normalize(0.5, 0.5)  # 数据归一化处理，将数据整理到[-1,1]之间；可让数据呈正态分布
    # ])
    #
    # # 下载数据到指定的文件夹
    # train_ds = torchvision.datasets.MNIST('ministdata',
    #                                       train=True,
    #                                       transform=transform,
    #                                       download=True)
    #
    # datalodar = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # gen = Generator().to(device)
    # dis = Discriminator().to(device)
    # gan=Gan()
    # 训练器的优化器
    # d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
    d_optim = torch.optim.Adam(dis.parameters(), lr=lr)
    # 训练生成器的优化器
    g_optim = torch.optim.Adam(gen.parameters(), lr=lr)

    # 交叉熵损失函数
    loss_fn = torch.nn.BCELoss()
    test_input = torch.randn(16, 100, device=device)  # 16个长度为100的随机数

    D_loss = []
    G_loss = []
    # 训练循环
    writer = SummaryWriter('gan_log')
    for epoch in range(EPOCH):
        # 初始化损失值
        d_epoch_loss = 0
        g_epoch_loss = 0
        # count = len(datalodar)  # 返回批次数
        count = len(labelled)  # 返回批次数
        # 对数据集进行迭代
        # for step, (img, _) in enumerate(datalodar):
        for step, (img, _) in enumerate(labelled):
            img = img.to(device)  # 把数据放到设备上
            img = torch.Tensor.float(img)
            size = img.size(0)  # img的第一位是size,获取批次的大小
            random_noise = torch.randn(size, 100, device=device)
            # gen_img, real_output, fake_output_d, fake_output=gan(img)

            # 判别器训练(真实图片的损失和生成图片的损失),损失的构建和优化
            d_optim.zero_grad()  # 梯度归零
            # 判别器对于真实图片产生的损失
            real_output = dis(img)  # 判别器输入真实的图片，real_output对真实图片的预测结果
            d_real_loss = loss_fn(real_output,
                                  torch.ones_like(real_output)
                                  )
            # d_real_loss.backward()  # 计算梯度

            # 在生成器上去计算生成器的损失，优化目标是判别器上的参数
            gen_img = gen(random_noise)  # 得到生成的图片
            # 因为优化目标是判别器，所以对生成器上的优化目标进行截断
            fake_output = dis(gen_img.detach())  # 判别器输入生成的图片，fake_output对生成图片的预测;detach会截断梯度，梯度就不会再传递到gen模型中了
            # 判别器在生成图像上产生的损失
            d_fake_loss = loss_fn(fake_output,
                                  torch.zeros_like(fake_output)
                                  )
            # d_fake_loss.backward()
            # 判别器损失
            d_loss = (d_real_loss + d_fake_loss)/2
            d_loss.backward()
            # 判别器优化
            d_optim.step()

            # 生成器上损失的构建和优化
            g_optim.zero_grad()  # 先将生成器上的梯度置零
            fake_output = dis(gen_img)
            g_loss = loss_fn(fake_output,
                             torch.ones_like(fake_output)
                             )  # 生成器损失
            g_loss.backward()
            g_optim.step()
            # 累计每一个批次的loss
            with torch.no_grad():
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss
        # 求平均损失

        with torch.no_grad():
            d_epoch_loss /= count
            g_epoch_loss /= count
            D_loss.append(d_epoch_loss)
            G_loss.append(g_epoch_loss)
            print('Epoch:', epoch,'d_loss:',d_epoch_loss,'g_loss:',g_epoch_loss)
            if (epoch+1) % 50 == 0:
                gen_img_plot(gen, test_input)
            # gen_img_plot(gen, test_input)
            writer.add_scalars("Discriminator_loss", {"train": np.round(d_epoch_loss.numpy(), 2)}, epoch + 1)
            writer.add_scalars("Generator", {"train": np.round(g_epoch_loss.numpy(), 2)}, epoch + 1)
    torch.save(gen.state_dict(), '../gan_classification/gen.pth')