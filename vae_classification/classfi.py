import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import vae_classification
from vae_classification.vae_utils import get_mnist
from vae_classification.vae_m2 import VAE,M2
from vae_classification.vae_utils import init_weights,training_class,test_class,training_VAE
import torch
from collections import defaultdict
from vae_classification.VAE import vae

from tensorboardX import SummaryWriter
from torchsummary import summary
labelled, test_set = get_mnist(batch_size=100, labels_per_class=10)

# fig, axs = plt.subplots(4, 4, figsize=(8,8))
# a=axs.flat
# for i, ax in enumerate(axs.flat):
#     ax.plot(reconstruction[i].cpu().detach().numpy())
#     ax.axis('off')
#
# plt.suptitle('Untrained model (initialized) output')
# plt.show()

beta = 0.9
alpha = 50
num_classes = 10
num_features = 1000  # 输出层维度
# num_features = 1  # 输入层通道
latent_f = 50  # 隐变量的空间维度 重点 单向GRU
# latent_f = 30
layers_enc = [num_features, 512, 256]  ##Adding extra layer
# layers_enc = [num_features,16,32, 64]
layers_dec = [latent_f + num_classes, 256, 512]
# layers_dec = [latent_f + num_classes, 32, 16,1]
# layers_dec_vae = [latent_f, 256, 512]
# in_channels=2
# layer_class = [1,1,1,1]
layer_class = [2,50,100,30]
# vae=VAE(layers_enc,layers_dec_vae,latent_f,num_features,beta)


# init_weights(vae) #神经网络需要初始化权重
# print(vae)
lr = 3.5e-3  #单向GRU
# lr = 1e-3
epoch = 0
# epoch_vae=0
# EPOCH_vae=50
EPOCH = 200
# optimizer_vae=torch.optim.Adam(vae.parameters(),lr=lr)

cuda = torch.cuda.is_available()
# cuda=False
print("Cuda is available :", cuda)
# if(cuda):
#     m2.cuda()
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# m2.to(device)
training_data = defaultdict(list)  # defaultdict作用是当key不存在时,返回的是工厂函数的默认值,比如list对应[ ],str对应的是空字符串
training_data_labelled = defaultdict(list)
training_data_unlabelled = defaultdict(list)
test_data = defaultdict(list)

# while epoch_vae<EPOCH_vae:
#     epoch_vae+=1;
#     print("\nvaeepoch",epoch_vae)
#     training_VAE(vae,labelled,optimizer_vae,cuda)
# torch.save(vae.state_dict(),'../vae_classification/vae.pth')


print('---------------------M2模型开始训练-----------------------------')
m2 = M2(layers_enc, layers_dec, latent_f, num_features, alpha, beta,  # vae
        layer_class, num_classes, dropout=0.5, activation="ReLU",  # vae
        activation_classifier='ReLU', dropout_classifier=0.5)  # extra vae
init_weights(m2)  # 神经网络需要初始化权重

# print(summary(m2,(2,500),batch_size=100))
optimizer = torch.optim.Adam(m2.parameters(), lr=lr)
if __name__ == '__main__':
    # vae.load_state_dict(torch.load('../vae_classification/vae_200_new_2.pth'))
    print(m2)
    writer = SummaryWriter('class_log_new_200_new-2')

    while epoch<EPOCH:
        epoch+=1
        print("\nepoch: ",epoch)
        training_class(m2,labelled,optimizer,cuda,training_data,training_data_labelled,training_data_unlabelled,writer,epoch)
        test_class(m2,test_set,alpha,cuda,test_data)

    writer.close()
    # print('123')
