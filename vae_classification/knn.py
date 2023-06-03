#从torchvision中引入常用数据集（MNIST），以及常用的预处理操作（transfrom）
from torchvision import datasets, transforms
#引入numpy计算矩阵
import numpy as np
#引入模型评估指标 accuracy_score
from sklearn.metrics import accuracy_score
import numpy
from sklearn.neighbors import KNeighborsClassifier
import torch
#引入进度条设置以及时间设置
from tqdm import tqdm
from vae_classification.vae_utils import get_mnist
import time

# 定义KNN函数
def KNN(train_x, train_y, test_x, test_y, k):
    #获取当前时间
    since = time.time()
    #可以将m,n理解为求其数据个数，属于torch.tensor类
    m = test_x.size(0)
    n = train_x.size(0)

    # 计算欧几里得距离矩阵，矩阵维度为m*n；
    print("计算距离矩阵")

    #test,train本身维度是m*1, **2为对每个元素平方，sum(dim=1，对行求和；keepdim =True时保持二维，
    # 而False对应一维，expand是改变维度，使其满足 m * n)
    xx = (test_x ** 2).sum(dim=1, keepdim=True).expand(m, n)
    #最后增添了转置操作
    yy = (train_x ** 2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)
    #计算近邻距离公式
    dist_mat = xx + yy - 2 * test_x.matmul(train_x.transpose(0, 1))
    #对距离进行排序
    mink_idxs = dist_mat.argsort(dim=-1)
    #定义一个空列表
    res = []
    for idxs in mink_idxs:
        # voting
        #代码下方会附上解释np.bincount()函数的博客
        res.append(np.bincount(np.array([train_y[idx] for idx in idxs[:k]])).argmax())

    assert len(res) == len(test_y)
    print("acc", accuracy_score(test_y, res))
    #计算运行时长
    time_elapsed = time.time() - since
    print('KNN mat training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#欧几里得距离计算公式
def cal_distance(x, y):
    return torch.sum((x - y) ** 2) ** 0.5
# KNN的迭代函数
def KNN_by_iter(train_x, train_y, test_x, test_y, k):
    since = time.time()

    # 计算距离
    res = []
    for x in tqdm(test_x):
        dists = []
        for y in train_x:
            dists.append(cal_distance(x, y).view(1))
        #torch.cat()用来拼接tensor
        idxs = torch.cat(dists).argsort()[:k]
        res.append(np.bincount(np.array([train_y[idx] for idx in idxs])).argmax())

    # print(res[:10])
    print("acc", accuracy_score(test_y, res))

    time_elapsed = time.time() - since
    print('KNN iter training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    # #加载数据
    # train_dataset = datasets.MNIST(root="./data", download=True, transform=transforms.ToTensor(), train=True)
    # test_dataset = datasets.MNIST(root="./data", download=True, transform=transforms.ToTensor(), train=False)
    #
    # # 组织训练，测试数据
    # train_x = []
    # train_y = []
    # for i in range(len(train_dataset)):
    #     img, target = train_dataset[i]
    #     train_x.append(img.view(-1))
    #     train_y.append(target)
    #
    #     if i > 5000:
    #         break
    #
    # # print(set(train_y))
    #
    # test_x = []
    # test_y = []
    # for i in range(len(test_dataset)):
    #     img, target = test_dataset[i]
    #     test_x.append(img.view(-1))
    #     test_y.append(target)
    #
    #     if i > 200:
    #         break
    labelled, test_set = get_mnist(batch_size=100, labels_per_class=10)
    train_features = []
    train_label = []
    test_features = []
    test_label = []



    # train_set = attDataset(path=path_train)
    # test_set = attDataset(path=path_test)

    for att, label in labelled:
        train_features.append(att)
        train_label.append(label)

    for att, label in test_set:
        test_features.append(att)
        test_label.append(label)
    train_features = train_features[0]
    train_label = train_label[0]
    test_features = test_features[0]
    test_label = test_label[0]
    # train_features = train_features.cpu().numpy()
    # train_label = numpy.array(train_label)
    # test_features = numpy.array(test_features)
    # test_label = numpy.array(test_label)
    # print("classes:", set(train_label))
    # train_x, train_y, test_x, test_y = load_data()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features, train_label)
    acc = knn.score(test_features, test_label)
    print('accuracy:', acc)
    # KNN(torch.stack(train_features), train_label, torch.stack(test_features), test_label, 7)
    # KNN(train_features, train_label, test_features, test_label, 7)
    # KNN_by_iter(train_features, train_label, test_features, test_label, 7)