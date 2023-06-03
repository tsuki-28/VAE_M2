import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import vae_classification
from vae_classification.vae_utils import get_mnist
# from vae_classification.vae_m2 import VAE,M2
from vae_classification.vae_utils import init_weights,training_M2,test_M2,training_VAE,training
import torch
from collections import defaultdict
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from tensorboardX import SummaryWriter
class Encoder(nn.Module):
    '''
    Encoder class that similar to FFNN class.
    '''

    def __init__(self, layers_, latent_features, activation="ReLU", batchnorm: bool = True, dropout: float = None):
        super().__init__()
        self.layers = []
        # layer construction
        # self.kernel_size=kernel_size
        # self.stride=stride
        for i in range(len(layers_) - 1):  # more than 2 layers
            self.layers.append(nn.Linear(layers_[i], layers_[i + 1]))
            # self.layers.append(nn.Conv2d(layers_[i], layers_[i + 1],kernel_size=(1,7),stride=(1,3)))
            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation == "Softplus":
                self.layers.append(nn.Softplus())
            elif activation=="Sigmoid":
                self.layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError("Wrong activation function")
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(layers_[i + 1])) #是否输出归一化维度
                # self.layers.append(nn.BatchNorm2d(layers_[i + 1]))  # 是否输出归一化维度
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        # finalize net
        self.net = nn.Sequential(*self.layers)

        self.mu_dense = torch.nn.Linear(layers_[-1], latent_features) #生成均值分布
        self.log_var_dense = nn.Linear(layers_[-1], latent_features) #生成方差分布

    def forward(self, x):
        x=x.view(x.size(0),-1)
        x = self.net(x)
        mu = self.mu_dense(x)
        log_var = self.log_var_dense(x)
        z = Normal(mu, (0.5 * log_var).exp()).rsample()
        # return tensor mu and sigma or return directly the distribution using the gaussian class
        return z, mu, log_var
        # return



# define network
class Decoder(nn.Module):
    '''
    Decoder class that similar to FFNN class.
    '''

    def __init__(self, layers_, num_output_, activation="ReLU", batchnorm=True, dropout: float = None):
        super().__init__()
        self.layers = []
        # self.kernel_size = kernel_size
        # self.stride = stride
        # layer construction
        for i in range(len(layers_) - 1):  # more than 2 layers
            self.layers.append(nn.Linear(layers_[i], layers_[i + 1]))
            # self.layers.append(nn.Conv2d(layers_[i], layers_[i + 1],kernel_size=(1,7),stride=(1,3)))
            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation == "Softplus":
                self.layers.append(nn.Softplus())
            elif activation=="Sigmoid":
                self.layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError("Wrong activation function")
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(layers_[i + 1]))
                # self.layers.append(nn.BatchNorm2d(layers_[i + 1]))
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        # output layer
        self.layers.append(nn.Linear(layers_[-1], num_output_))

        # finalize net
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)
class VAE(nn.Module):
    '''
    Beta Variational Auto-Encoder that is built upon the Encoder-Decor class;
    The initialization is the same as the FFNN class, moreover the Reconstruction loss is chosen to be "Binary Cross Entropy"
    and the KL divergence is computed in analytical form (both prior and posterior are Gaussion)
    '''

    def __init__(self, enc_layers, dec_layers, latent_features, num_output,beta=1.0,activation="ReLU",
                 batchnorm: bool = True, dropout: float = None):

        super().__init__()
        self.beta = beta
        self.encoder = Encoder(enc_layers, latent_features, activation, batchnorm, dropout)
        self.decoder = Decoder(dec_layers, num_output, activation, batchnorm, dropout)

    def encode(self, x):
        # encoding from z-batch, must return 2 parameter [mu,sigma],
        z, mu, log_var = self.encoder(x)
        # extract mu and sigma from encoder result
        # mu = self.mu_dense(x)
        # log_var = self.log_var_dense(x)
        # return tensor mu and sigma or return directly the distribution using the gaussian class
        return z, mu, log_var

    def decode(self, z):
        # decoding
        z = self.decoder(z)
        z = torch.sigmoid(z)
        return z

    def sample(self, n: int, z=None):
        # generation of image = sampling from random noise + decode

        if not z:
            return self.decode(torch.randn(n, self.latent_features))
        else:
            return self.decode(z)

    def elbo(self, x, mu, log_var, z, rec):
        # loss function = reconstruction error + KL-divergence
        x=x.view(x.size(0),-1)
        BCE_new = - torch.sum(F.binary_cross_entropy(rec, x, reduction="none"), dim=1)  # mean /none
        KL_analyt = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)  # analytical KL

        # ELBO
        ELBO = BCE_new - self.beta * KL_analyt
        # ELBO = BCE - self.beta * KL

        loss = -ELBO.mean()

        with torch.no_grad():
            diagnostics = {'elbo': -ELBO.mean(), 'likelihood': BCE_new, 'KL': KL_analyt}
            # diagnostics = {'elbo': ELBO, 'likelihood': BCE, 'KL': KL}

        return loss, diagnostics

    def forward(self, x):
        # posterior param
        z, mu, log_var = self.encode(x)

        # posterior dist (force sigma to be positive with log) + reparametrization
        # post_dist = Normal(mu, (0.5*log_var).exp())
        # z = post_dist.rsample()

        # reconstruction -> log prob with sigmoid
        rec = self.decode(z)

        # ELBO
        loss, diagnostic = self.elbo(x, mu, log_var, z, rec)

        return [loss, diagnostic, z, rec]
labelled,test_set = get_mnist(batch_size=100,labels_per_class=10)
beta=1
alpha=50
num_classes=10
num_features = 1000 #输出层维度
# num_features = 2  # 输入层通道
latent_f =80#隐变量的空间维度 重点
layers_enc = [num_features,512,256]  ##Adding extra layer
layers_dec = [latent_f+num_classes,256,512]
layers_dec_vae = [latent_f, 256,512]
# layer_class=[num_features,300,120]
vae=VAE(layers_enc,layers_dec_vae,latent_f,num_features,beta)



init_weights(vae)  # 神经网络需要初始化权重
print(vae)
lr = 3e-4
# epoch=0
epoch_vae = 0
EPOCH_vae = 200
# EPOCH=200
optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr)

cuda = torch.cuda.is_available()
# cuda=False
print("Cuda is available :", cuda)
if __name__ == '__main__':

    writer = SummaryWriter('vae_log_antenna2')
    # if(cuda):
    #     m2.cuda()
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # m2.to(device)
    training_data=defaultdict(list)   #defaultdict作用是当key不存在时,返回的是工厂函数的默认值,比如list对应[ ],str对应的是空字符串
    training_data_labelled=defaultdict(list)
    training_data_unlabelled=defaultdict(list)
    test_data=defaultdict(list)
    print('---------------------VAE模型开始训练-----------------------------')

    while epoch_vae<EPOCH_vae:
        epoch_vae+=1
        print("\nvaeepoch",epoch_vae)
        training(vae,labelled,optimizer_vae,cuda,training_data,writer,epoch_vae)
    torch.save(vae.state_dict(),'../vae_classification/vae_antenna2.pth')
    writer.close()
    # vae.load_state_dict(torch.load('../vae_classification/vae_200_new.pth'))
    # for input, target in labelled:
    #     with torch.no_grad():
    #         input=torch.Tensor.float(input)
    #         _,_,_,rec_1=vae(input)
    #         input=input.view(input.size(0),2,500)
    #         rec_1 = rec_1.view(rec_1.size(0), 2, 500)
    #         rec_1=rec_1.detach().numpy()
    #         input=input.numpy()
    #         fig,axs=plt.subplots(4,4,figsize=(8,8))
    #         a=axs.flat
    #         fig_re, axs_re = plt.subplots(4, 4, figsize=(8, 8))
    #         a_re = axs_re.flat
    #         for i,ax in enumerate(axs.flat):
    #             ax.plot(input[i,0])
    #             ax.plot(input[i, 1])
    #             path="../vae_classification/orignal"
    #             name = str(i) + '.csv'
    #             path=os.path.join(path,name)
    #             pd.DataFrame(input[i]).round(5).to_csv(path, header=False, index=False)
    #             # ax.axis('off')
    #         plt.suptitle('orignal')
    #         for i,ax in enumerate(axs_re.flat):
    #             ax.plot(rec_1[i,0])
    #             ax.plot(rec_1[i,1])
    #             # ax.axis('off')
    #             ax.title.set_text(str(i))
    #             path = "../vae_classification/rebuild"
    #             name=str(i)+'.csv'
    #             path = os.path.join(path, name)
    #             pd.DataFrame(rec_1[i]).round(5).to_csv(path, header=False, index=False)
    #         plt.suptitle('rebuild')
    #         plt.show()
    #     break

    # vae.load_state_dict(torch.load('vae.path'))
    print('---------------------VAE模型训练结束-----------------------------')
    # print('---------------------M2模型开始训练-----------------------------')
    # m2 = M2(layers_enc, layers_dec, latent_f, num_features, alpha, beta,  # vae
    #         layer_class, num_classes, dropout=0.3, activation="ReLU",  # vae
    #         activation_classifier='ReLU', dropout_classifier=0.3)  # extra vae
    # init_weights(m2)  # 神经网络需要初始化权重
    # print(m2)
    # optimizer = torch.optim.Adam(m2.parameters(), lr=lr)
    # while epoch<EPOCH:
    #     epoch+=1
    #     print("\nepoch: ",epoch)
    #     training_M2(m2,vae,labelled,optimizer,cuda,training_data,training_data_labelled,training_data_unlabelled)
    #     test_M2(m2,test_set,alpha,cuda,test_data)
    #

    # print('123')
