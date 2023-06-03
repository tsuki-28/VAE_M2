import os

from scipy.misc import derivative

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import scipy
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import random
from torch.utils.data import Subset
import torch.nn.functional as F
import lttb
import scipy.signal as signal
from shutil import copy2

class myDataSet(Dataset):
    def __init__(self,label_dir, transform=None):
        """
        :param label_dir: 标签文件路径
        :param transform: transform操作
        """
        self.transform = transform
        self.data_name=[]
        self.label_name=[]
        # self.label_index=[]
        # 读文件夹下每个数据文件名称
        # os.listdir读取文件夹内的文件名称
        # self.file_name = os.listdir(data_dir)
        # 读取标签文件的标签名
        self.label_fname = os.listdir(label_dir)
        # 读取每个标签下的数据
        # 以及为每个数据创建标签
        for i in range(len(self.label_fname)):
            data_dir=os.path.join(label_dir,self.label_fname[i])
            temp_data=os.listdir(data_dir)
            self.data_name.extend(temp_data)
            # self.label_index.extend(i for j in range(len(temp_data)))
            self.label_name.extend(self.label_fname[i] for j in range(len(temp_data)))
        self.data_path = []
        self.labels = []
        self.target=[]
        # 让每一个文件的路径拼接起来
        for index in range(len(self.data_name)):
            self.data_path.append(os.path.join(os.path.join(label_dir, self.label_name[index]), self.data_name[index]))
            self.labels.append(self.label_fname.index(self.label_name[index]))
        self.labels=torch.tensor(self.labels)
        self.target=self.labels
    def __len__(self):
        # 返回数据集长度
        return len(self.data_name)

    def __getitem__(self, index):
        # 获取每一个数据
        # print(index)
        # print(self.label_name[index])
        # 读取数据
        name=self.data_path[index]
        beforData = pd.read_csv(self.data_path[index], header=None)
        data=self.dataProcess_2(beforData)
        # data=data.reshape(29600)
        # 读取标签
        label = self.label_fname.index(self.label_name[index])


        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        # 转成张量
        data = torch.tensor(data)

        label = torch.tensor(label)

        return data, label # 返回数据和标签
        # return data, label,self.label_name[index],self.data_name[index]  # 返回数据和标签
    def dataProcess(self,A):
        A=A.to_numpy()
        A[:,3]= A[:,3]*(2*math.pi/4096)
        phase=[]
        phase_ref=[]
        rss=[]
        rss_ref=[]
        time = []
        time_ref = []
        rss_2 = []
        rss_ref_2 = []
        phase_2 = []
        phase_ref_2 = []
        time_2 = []
        time_ref_2 = []
        # A_process=[]
        # A_process=np.ones((2,500,2))
        A_process = np.ones((2,500))
        temp_B=[]
        temp_C=[]
        for i in range(len(A)):
            if(A[i,0]==1107):
                if(A[i,1]==1):
                    phase.append(A[i,3])
                    time.append(A[i,6])
                    rss.append(A[i,4])
                elif(A[i,1]==2):
                    phase_2.append(A[i,3])
                    time_2.append(A[i,6])
                    rss_2.append(A[i,4])
            elif (A[i,0] == 1208):
                if (A[i,1] == 1):
                    phase_ref.append(A[i,3])
                    time_ref.append(A[i,6])
                    rss_ref.append(A[i,4])
                elif (A[i,1]== 2):
                    phase_ref_2.append(A[i,3])
                    time_ref_2.append(A[i,6])
                    rss_ref_2.append(A[i,4])
        for i in range(len(phase)-1):
            while phase[i+1]-phase[i]>2:
                phase[i+1]=phase[i+1]-math.pi
            while phase[i]-phase[i+1]>2:
                phase[i+1]=phase[i+1]+math.pi
        for i in range(len(phase_2)-1):
            while phase_2[i+1]-phase_2[i]>2:
                phase_2[i+1]=phase_2[i+1]-math.pi
            while phase_2[i]-phase_2[i+1]>2:
                phase_2[i+1]=phase_2[i+1]+math.pi
        for i in range(len(phase_ref)-1):
            while phase_ref[i+1]-phase_ref[i]>2:
                phase_ref[i+1]=phase_ref[i+1]-math.pi
            while phase_ref[i]-phase_ref[i+1]>2:
                phase_ref[i+1]=phase_ref[i+1]+math.pi
        for i in range(len(phase_ref_2)-1):
            while phase_ref_2[i+1]-phase_ref_2[i]>2:
                phase_ref_2[i+1]=phase_ref_2[i+1]-math.pi
            while phase_ref_2[i]-phase_ref_2[i+1]>2:
                phase_ref_2[i+1]=phase_ref_2[i+1]+math.pi
        # plt.suptitle('orignal')
        # plt.subplot(211)
        # x1=(time - time[0])/ 1000000
        # plt.plot(x1,phase, linewidth=0.5)
        # plt.subplot(212)
        # x2=(time_2-time_2[0])/1000000
        # plt.plot(x2,phase_2, linewidth=0.5)
        # plt.show()
        phase=self.calculateKalmanFilter(phase)
        phase_2=self.calculateKalmanFilter(phase_2)
        phase_ref=self.calculateKalmanFilter(phase_ref)
        phase_ref_2=self.calculateKalmanFilter(phase_ref_2)
        phase = signal.medfilt(phase,21)
        phase_2 = signal.medfilt(phase_2,21)
        phase_ref = signal.medfilt(phase_ref,21)
        phase_ref_2 = signal.medfilt(phase_ref_2,21)
        # plt.suptitle('filter')
        # plt.subplot(211)
        # x1 = (time - time[0]) / 1000000
        # plt.plot(x1, phase, linewidth=0.5)
        # plt.subplot(212)
        # x2 = (time_ref - time_ref[0]) / 1000000
        # plt.plot(x2, phase_2, linewidth=0.5)
        # plt.show()
        time1,r1=np.unique(time,return_index=True)
        for i in range(len(phase)-1,-1,-1):
            if ~np.isin(i,r1):
                phase=np.delete(phase,i)
        time2,r2= np.unique(time_ref,return_index=True)
        for i in range(len(phase_ref)-1,-1,-1):
            if ~np.isin(i, r2):
                phase_ref = np.delete(phase_ref, i)
        time1_2,r1_2= np.unique(time_2,return_index=True)
        for i in range(len(phase_2)-1,-1,-1):
            if ~np.isin(i, r1_2):
                phase_2 = np.delete(phase_2, i)
        time2_2,r2_2= np.unique(time_ref_2,return_index=True)
        for i in range(len(phase_ref_2)-1,-1,-1):
            if ~np.isin(i, r2_2):
                phase_ref_2 = np.delete(phase_ref_2, i)
        time_ans=np.sort(np.unique(np.hstack([time1,time2])))
        phase_yx=scipy.interpolate.PchipInterpolator(time1,phase,extrapolate=0)
        phase=phase_yx(time_ans)
        phase_ref_yx=scipy.interpolate.PchipInterpolator(time2,phase_ref,extrapolate=0)
        phase_ref=phase_ref_yx(time_ans)
        time_ans_2=np.sort(np.unique(np.hstack([time1_2,time2_2])))
        phase_2_yx=scipy.interpolate.PchipInterpolator(time1_2,phase_2,extrapolate=0)
        phase_2=phase_2_yx(time_ans_2)
        phase_ref_2_yx=scipy.interpolate.PchipInterpolator(time2_2,phase_ref_2,extrapolate=0)
        phase_ref_2=phase_ref_2_yx(time_ans_2)
        time_ans=time_ans-time_ans[0]
        s1=max(time1[0],time2[0])
        index_1=np.where(time_ans==s1)
        phase=np.delete(phase,index_1)
        phase_ref=np.delete(phase_ref,index_1)
        time_ans=np.delete(time_ans,index_1)
        time_ans=time_ans/1000000
        time_ans_2=time_ans_2-time_ans_2[0]
        s2=max(time1_2[0],time2_2[0])
        index_2=np.where(time_ans_2==s2)
        phase=np.delete(phase,index_2)
        phase_ref=np.delete(phase_ref,index_2)
        time_ans_2=np.delete(time_ans_2,index_2)
        time_ans_2=time_ans_2/1000000
        # for i in range(len(index_1)):
        K=phase-phase_ref
        K_2=phase_2-phase_ref_2
        for i in range(len(K)-1):
            while(K[i+1]-K[i]>2):
                K[i+1]=K[i+1]-math.pi
            while(K[i]-K[i+1]>2):
                K[i+1]=K[i+1]+math.pi
        for i in range(len(K_2) - 1):
            while (K_2[i + 1] - K_2[i] > 2):
                K_2[i + 1] = K_2[i + 1] - math.pi
            while (K_2[i] - K_2[i + 1] > 2):
                K_2[i + 1] = K_2[i + 1] + math.pi
        time_ans=time_ans[4:len(time_ans)-5]
        K=K[4:len(K)-5]
        time_ans_2=time_ans_2[4:len(time_ans_2)-5]
        K_2=K_2[4:len(K_2)-5]
        time_x=np.sort(np.unique(np.hstack([time_ans,time_ans_2])))
        ##去除nan值
        if np.isnan(np.min(K)):
            bool_nan_1=np.isnan(K)
            index_nan_1=np.where(bool_nan_1)
            a=index_nan_1[0]
            K=np.delete(K,a)
            time_ans=np.delete(time_ans,a)
            # for i in range(len(index_nan_1)-1,-1,-1):
            #     np.delete(K,index_nan_1[i])
            #     np.delete(time_ans,index_nan_1[i])
        if np.isnan(np.min(K_2)):
            bool_nan_2=np.isnan(K_2)
            index_nan_2=np.where(bool_nan_2)
            b=index_nan_2[0]
            K_2=np.delete(K_2,b)
            time_ans_2=np.delete(time_ans_2,b)
        # phase_fy1=scipy.interpolate.PchipInterpolator(time_ans,K,extrapolate=0)
        # phase_fy2=scipy.interpolate.PchipInterpolator(time_ans_2,K_2,extrapolate=0)
        # phase_y1=phase_fy1(time_x)
        # phase_y2=phase_fy2(time_x)
        # # phase_y=phase_y1-phase_y2
        # # A_process=phase_y
        # if np.isnan(np.min(phase_y1)):
        #     bool_nan=np.isnan(phase_y1)
        #     index_nan=np.where(bool_nan)
        #     b=index_nan[0]
        #     phase_y1=np.delete(phase_y1,b)
        # pad_value_1=phase_y1[len(phase_y1)-1]
        # phase_y1=np.pad(phase_y1,pad_width=(0,14800-phase_y1.shape[0]),mode='constant',constant_values=(0,pad_value_1))
        # if np.isnan(np.min(phase_y2)):
        #     bool_nan=np.isnan(phase_y2)
        #     index_nan=np.where(bool_nan)
        #     b=index_nan[0]
        #     phase_y2=np.delete(phase_y2,b)
        # pad_value_2=phase_y2[len(phase_y2)-1]
        # phase_y2=np.pad(phase_y2,pad_width=(0,14800-phase_y2.shape[0]),mode='constant',constant_values=(0,pad_value_2))
        # phase_y2=phase_y2-(phase_y2[0]-phase_y1[len(phase_y1-1)])
        phase_y1 =K
        # phase_y2 = K_2
        phase_y2 =K_2-(K_2[0]-K[len(K)-1])
        time_ans=time_ans[:,np.newaxis]
        time_ans_2=time_ans_2[:,np.newaxis]
        phase_y1=phase_y1[:,np.newaxis]
        phase_y2=phase_y2[:,np.newaxis]
        '''
        stft变化
          # aa = []
        # for i in range(200):
        #     aa.append(np.sin(0.3 * np.pi * i))
        # for i in range(200):
        #     aa.append(np.sin(0.13 * np.pi * i))
        # for i in range(200):
        #     aa.append(np.sin(0.05 * np.pi * i))
        # fs=2
        # f,t,nd=signal.stft(phase_y1,fs=fs,window='hann',nperseg=150,noverlap=0)
        # plt.pcolormesh(t,f,np.abs(nd), vmin = 0, vmax = 4)
        # plt.title('STFT')
        # plt.ylabel('frequency')
        # plt.xlabel('time')
        # plt.show()
        '''

        data1=np.hstack([time_ans,phase_y1])
        # pd.DataFrame(data1).round(5).to_csv('./1_lttb_old.csv', header=False, index=False)
        data2=np.hstack([time_ans_2,phase_y2])
        data1=lttb.downsample(data1,n_out=500)
        data2=lttb.downsample(data2,n_out=500)
        # plt.plot(time_ans_2,phase_y2)
        # plt.plot(data2[:, 0],data2[:, 1])
        # data2_normal=self.Zscore(data2[:, 1])
        data2_normal = self.normalize(data2[:, 1])
        # data2_normal = self.Zscore(data2_normal)
        # plt.plot(data2[:, 0],data2_normal)

        # plt.plot(time_ans,phase_y1)
        # plt.plot(data1[:, 0],data1[:, 1])
        # data1_normal=self.Zscore(data1[:, 1])
        data1_normal = self.normalize(data1[:, 1])
        # data1_normal = self.Zscore(data1_normal)
        # plt.plot(data1[:, 0],data1_normal)
        # plt.show()
        # phase_y1=torch.tensor(phase_y1)
        # phase_y2=torch.tensor(phase_y2)
        # phase_y1_rsa=F.interpolate(phase_y1,scale_factor=0.2)
        # phase_y2_rsa=F.interpolate(phase_y2,scale_factor=0.2)
        # A_process[0,:]=data1[:,1]
        # A_process[1, :] = data2[:, 1]
        # A_process[0,:,:]=data1[]
        # A_process=np.expand_dims(A_process,0)
        # A_process[1,:,:]=data2
        # data2_normal=data2_normal-(np.mean(data2_normal)-np.mean(data1_normal))
        # data2_normal = data2_normal-(np.mean(data1_normal)-data2_normal[0])
        A_process[0,:]=data1_normal
        A_process[1,:] = data2_normal

        # A_process = np.hstack([data1_normal,data2_normal])
        # if np.isnan(np.min(A_process)):
        #     bool_nan=np.isnan(A_process)
        #     index_nan=np.where(bool_nan)
        #     b=index_nan[0]
        #     A_process=np.delete(A_process,b)
        # A_process.append(phase_y1)
        # A_process.append(phase_y2)
        # A_process.extend(phase_y2)
        # pad_value=A_process[len(A_process)-1]
        # A_process = np.pad(A_process, pad_width=(0, 14800 - len(A_process)), mode='constant',
        #                   constant_values=(0, pad_value))
        # A_process = np.pad(A_process, pad_width=(0, 2960 - len(A_process)), mode='constant',
        #                  constant_values=(0, pad_value))
        # A_process=np.array(A_process)
        # phase_y=self.calculateKalmanFilter(phase_y)
        # plt.plot(data1_normal)
        # plt.plot(data2_normal)
        # plt.plot(A_process)
        # plt.plot(A_process[1])
        # plt.show()
        # antenna1=np.ones(len(time_ans))
        # antenna2=np.ones(len(time_ans_2))*2
        # temp_B.append(antenna1[0:])
        # temp_B.append(time_ans)
        # temp_B.append(K)
        # temp_C.append(antenna2)
        # temp_C.append(time_ans_2)
        # temp_C.append(K_2)
        # A_process=np.hstack([temp_B,temp_C])
        # A_process.append(time_ans)
        # A_process.append(K)
        # A_process.append(time_ans_2)
        # A_process.append(K_2)
        # plt.suptitle('final')
        # plt.subplot(211)
        # plt.plot(K,linewidth=0.5)
        # plt.subplot(212)
        # plt.plot(K_2,linewidth=0.5)
        # plt.show()
        # pd.DataFrame(data1).round(5).to_csv('./1_lttb.csv', header=False, index=False)
        # pd.DataFrame(data2).round(5).to_csv('./2_lttb.csv', header=False, index=False)
        # return data1,data2
        return A_process
    def dataProcess_2(self,A):
        A=A.to_numpy()
        A_process = np.ones((2,500))
        # A_process[0,:]=A[0,:]
        # A_process[1,:]=A[1,:]
        # A_process=np.expand_dims(A_process,0)
        # plt.plot(A_process[0,0,:])
        # plt.plot(A_process[0,1,:])
        # plt.show()
        A_process = np.hstack([A[0,:], A[1,:]])
        # A_process = A[1, :]
        return A_process

    def dataProcess_nolttb(self,A):
        A=A.to_numpy()
        A[:,3]= A[:,3]*(2*math.pi/4096)
        phase=[]
        phase_ref=[]
        rss=[]
        rss_ref=[]
        time = []
        time_ref = []
        rss_2 = []
        rss_ref_2 = []
        phase_2 = []
        phase_ref_2 = []
        time_2 = []
        time_ref_2 = []
        A_process=[]
        # A_process=np.ones((2,500,2))
        # A_process = np.ones((2,500))
        for i in range(len(A)):
            if(A[i,0]==1107):
                if(A[i,1]==1):
                    phase.append(A[i,3])
                    time.append(A[i,6])
                    rss.append(A[i,4])
                elif(A[i,1]==2):
                    phase_2.append(A[i,3])
                    time_2.append(A[i,6])
                    rss_2.append(A[i,4])
            elif (A[i,0] == 1208):
                if (A[i,1] == 1):
                    phase_ref.append(A[i,3])
                    time_ref.append(A[i,6])
                    rss_ref.append(A[i,4])
                elif (A[i,1]== 2):
                    phase_ref_2.append(A[i,3])
                    time_ref_2.append(A[i,6])
                    rss_ref_2.append(A[i,4])
        for i in range(len(phase)-1):
            while phase[i+1]-phase[i]>2:
                phase[i+1]=phase[i+1]-math.pi
            while phase[i]-phase[i+1]>2:
                phase[i+1]=phase[i+1]+math.pi
        for i in range(len(phase_2)-1):
            while phase_2[i+1]-phase_2[i]>2:
                phase_2[i+1]=phase_2[i+1]-math.pi
            while phase_2[i]-phase_2[i+1]>2:
                phase_2[i+1]=phase_2[i+1]+math.pi
        for i in range(len(phase_ref)-1):
            while phase_ref[i+1]-phase_ref[i]>2:
                phase_ref[i+1]=phase_ref[i+1]-math.pi
            while phase_ref[i]-phase_ref[i+1]>2:
                phase_ref[i+1]=phase_ref[i+1]+math.pi
        for i in range(len(phase_ref_2)-1):
            while phase_ref_2[i+1]-phase_ref_2[i]>2:
                phase_ref_2[i+1]=phase_ref_2[i+1]-math.pi
            while phase_ref_2[i]-phase_ref_2[i+1]>2:
                phase_ref_2[i+1]=phase_ref_2[i+1]+math.pi
        # plt.suptitle('orignal')
        # plt.subplot(211)
        # x1=(time - time[0])/ 1000000
        # plt.plot(x1,phase, linewidth=0.5)
        # plt.subplot(212)
        # x2=(time_2-time_2[0])/1000000
        # plt.plot(x2,phase_2, linewidth=0.5)
        # plt.show()
        phase=self.calculateKalmanFilter(phase)
        phase_2=self.calculateKalmanFilter(phase_2)
        phase_ref=self.calculateKalmanFilter(phase_ref)
        phase_ref_2=self.calculateKalmanFilter(phase_ref_2)
        phase = signal.medfilt(phase,21)
        phase_2 = signal.medfilt(phase_2,21)
        phase_ref = signal.medfilt(phase_ref,21)
        phase_ref_2 = signal.medfilt(phase_ref_2,21)
        # plt.suptitle('filter')
        # plt.subplot(211)
        # x1 = (time - time[0]) / 1000000
        # plt.plot(x1, phase, linewidth=0.5)
        # plt.subplot(212)
        # x2 = (time_ref - time_ref[0]) / 1000000
        # plt.plot(x2, phase_2, linewidth=0.5)
        # plt.show()
        time1,r1=np.unique(time,return_index=True)
        for i in range(len(phase)-1,-1,-1):
            if ~np.isin(i,r1):
                phase=np.delete(phase,i)
        time2,r2= np.unique(time_ref,return_index=True)
        for i in range(len(phase_ref)-1,-1,-1):
            if ~np.isin(i, r2):
                phase_ref = np.delete(phase_ref, i)
        time1_2,r1_2= np.unique(time_2,return_index=True)
        for i in range(len(phase_2)-1,-1,-1):
            if ~np.isin(i, r1_2):
                phase_2 = np.delete(phase_2, i)
        time2_2,r2_2= np.unique(time_ref_2,return_index=True)
        for i in range(len(phase_ref_2)-1,-1,-1):
            if ~np.isin(i, r2_2):
                phase_ref_2 = np.delete(phase_ref_2, i)
        time_ans=np.sort(np.unique(np.hstack([time1,time2])))
        phase_yx=scipy.interpolate.PchipInterpolator(time1,phase,extrapolate=0)
        phase=phase_yx(time_ans)
        phase_ref_yx=scipy.interpolate.PchipInterpolator(time2,phase_ref,extrapolate=0)
        phase_ref=phase_ref_yx(time_ans)
        time_ans_2=np.sort(np.unique(np.hstack([time1_2,time2_2])))
        phase_2_yx=scipy.interpolate.PchipInterpolator(time1_2,phase_2,extrapolate=0)
        phase_2=phase_2_yx(time_ans_2)
        phase_ref_2_yx=scipy.interpolate.PchipInterpolator(time2_2,phase_ref_2,extrapolate=0)
        phase_ref_2=phase_ref_2_yx(time_ans_2)
        time_ans=time_ans-time_ans[0]
        s1=max(time1[0],time2[0])
        index_1=np.where(time_ans==s1)
        phase=np.delete(phase,index_1)
        phase_ref=np.delete(phase_ref,index_1)
        time_ans=np.delete(time_ans,index_1)
        time_ans=time_ans/1000000
        time_ans_2=time_ans_2-time_ans_2[0]
        s2=max(time1_2[0],time2_2[0])
        index_2=np.where(time_ans_2==s2)
        phase=np.delete(phase,index_2)
        phase_ref=np.delete(phase_ref,index_2)
        time_ans_2=np.delete(time_ans_2,index_2)
        time_ans_2=time_ans_2/1000000
        # for i in range(len(index_1)):
        K=phase-phase_ref
        K_2=phase_2-phase_ref_2
        for i in range(len(K)-1):
            while(K[i+1]-K[i]>2):
                K[i+1]=K[i+1]-math.pi
            while(K[i]-K[i+1]>2):
                K[i+1]=K[i+1]+math.pi
        for i in range(len(K_2) - 1):
            while (K_2[i + 1] - K_2[i] > 2):
                K_2[i + 1] = K_2[i + 1] - math.pi
            while (K_2[i] - K_2[i + 1] > 2):
                K_2[i + 1] = K_2[i + 1] + math.pi
        time_ans=time_ans[4:len(time_ans)-5]
        K=K[4:len(K)-5]
        time_ans_2=time_ans_2[4:len(time_ans_2)-5]
        K_2=K_2[4:len(K_2)-5]
        time_x=np.sort(np.unique(np.hstack([time_ans,time_ans_2])))
        ##去除nan值
        if np.isnan(np.min(K)):
            bool_nan_1=np.isnan(K)
            index_nan_1=np.where(bool_nan_1)
            a=index_nan_1[0]
            K=np.delete(K,a)
            time_ans=np.delete(time_ans,a)
            # for i in range(len(index_nan_1)-1,-1,-1):
            #     np.delete(K,index_nan_1[i])
            #     np.delete(time_ans,index_nan_1[i])
        if np.isnan(np.min(K_2)):
            bool_nan_2=np.isnan(K_2)
            index_nan_2=np.where(bool_nan_2)
            b=index_nan_2[0]
            K_2=np.delete(K_2,b)
            time_ans_2=np.delete(time_ans_2,b)
        # phase_fy1=scipy.interpolate.PchipInterpolator(time_ans,K,extrapolate=0)
        # phase_fy2=scipy.interpolate.PchipInterpolator(time_ans_2,K_2,extrapolate=0)
        # phase_y1=phase_fy1(time_x)
        # phase_y2=phase_fy2(time_x)
        # # phase_y=phase_y1-phase_y2
        # # A_process=phase_y
        phase_y1 =K
        phase_y2 = K_2
        if np.isnan(np.min(phase_y1)):
            bool_nan=np.isnan(phase_y1)
            index_nan=np.where(bool_nan)
            b=index_nan[0]
            phase_y1=np.delete(phase_y1,b)
        pad_value_1=phase_y1[len(phase_y1)-1]
        l2=len(phase_y2)
        l1=len(phase_y1)
        max_length=np.max([l1,l2])
        phase_y1=np.pad(phase_y1,pad_width=(0,7400-phase_y1.shape[0]),mode='constant',constant_values=(0,pad_value_1))
        if np.isnan(np.min(phase_y2)):
            bool_nan=np.isnan(phase_y2)
            index_nan=np.where(bool_nan)
            b=index_nan[0]
            phase_y2=np.delete(phase_y2,b)
        pad_value_2=phase_y2[len(phase_y2)-1]
        phase_y2=np.pad(phase_y2,pad_width=(0,7400-phase_y2.shape[0]),mode='constant',constant_values=(0,pad_value_2))
        # phase_y2=phase_y2-(phase_y2[0]-phase_y1[len(phase_y1-1)])
        # phase_y1 =K
        # phase_y2 = K_2
        # phase_y2 =K_2-(K_2[0]-K[len(K)-1])
        # time_ans=time_ans[:,np.newaxis]
        # time_ans_2=time_ans_2[:,np.newaxis]
        # phase_y1=phase_y1[:,np.newaxis]
        # phase_y2=phase_y2[:,np.newaxis]
        '''
        stft变化
          # aa = []
        # for i in range(200):
        #     aa.append(np.sin(0.3 * np.pi * i))
        # for i in range(200):
        #     aa.append(np.sin(0.13 * np.pi * i))
        # for i in range(200):
        #     aa.append(np.sin(0.05 * np.pi * i))
        # fs=2
        # f,t,nd=signal.stft(phase_y1,fs=fs,window='hann',nperseg=150,noverlap=0)
        # plt.pcolormesh(t,f,np.abs(nd), vmin = 0, vmax = 4)
        # plt.title('STFT')
        # plt.ylabel('frequency')
        # plt.xlabel('time')
        # plt.show()
        '''

        # data1=np.hstack([time_ans,phase_y1])
        # pd.DataFrame(data1).round(5).to_csv('./1_lttb_old.csv', header=False, index=False)
        # data2=np.hstack([time_ans_2,phase_y2])
        # data1=lttb.downsample(data1,n_out=500)
        # data2=lttb.downsample(data2,n_out=500)
        # plt.plot(time_ans_2,phase_y2)
        # plt.plot(data2[:, 0],data2[:, 1])
        # data2_normal=self.Zscore(data2[:, 1])
        # data2_normal = self.normalize(data2[:, 1])
        # data2_normal = self.Zscore(data2_normal)
        # plt.plot(data2[:, 0],data2_normal)

        # plt.plot(time_ans,phase_y1)
        # plt.plot(data1[:, 0],data1[:, 1])
        # data1_normal=self.Zscore(data1[:, 1])
        # data1_normal = self.normalize(data1[:, 1])
        # data1_normal = self.Zscore(data1_normal)
        # plt.plot(data1[:, 0],data1_normal)
        # plt.show()
        # phase_y1=torch.tensor(phase_y1)
        # phase_y2=torch.tensor(phase_y2)
        # phase_y1_rsa=F.interpolate(phase_y1,scale_factor=0.2)
        # phase_y2_rsa=F.interpolate(phase_y2,scale_factor=0.2)
        # A_process[0,:]=data1[:,1]
        # A_process[1, :] = data2[:, 1]
        # A_process[0,:,:]=data1[]
        # A_process=np.expand_dims(A_process,0)
        # A_process[1,:,:]=data2
        # data2_normal=data2_normal-(np.mean(data2_normal)-np.mean(data1_normal))
        # data2_normal = data2_normal-(np.mean(data1_normal)-data2_normal[0])
        # A_process[0,:]=data1_normal
        # A_process[1,:] = data2_normal

        # A_process = np.hstack([data1_normal,data2_normal])
        # if np.isnan(np.min(A_process)):
        #     bool_nan=np.isnan(A_process)
        #     index_nan=np.where(bool_nan)
        #     b=index_nan[0]
        #     A_process=np.delete(A_process,b)
        phase_y1=self.normalize(phase_y1)
        phase_y2=self.normalize(phase_y2)
        A_process.append(phase_y1)
        A_process.append(phase_y2)
        # A_process.extend(phase_y2)
        # pad_value=A_process[len(A_process)-1]
        # A_process = np.pad(A_process, pad_width=(0, 14800 - len(A_process)), mode='constant',
        #                   constant_values=(0, pad_value))
        # A_process = np.pad(A_process, pad_width=(0, 2960 - len(A_process)), mode='constant',
        #                  constant_values=(0, pad_value))
        A_process=np.array(A_process)
        # phase_y=self.calculateKalmanFilter(phase_y)
        # plt.plot(data1_normal)
        # plt.plot(data2_normal)
        # plt.plot(A_process[0,:])
        # plt.plot(A_process[1,:])
        # plt.show()
        # antenna1=np.ones(len(time_ans))
        # antenna2=np.ones(len(time_ans_2))*2
        # temp_B.append(antenna1[0:])
        # temp_B.append(time_ans)
        # temp_B.append(K)
        # temp_C.append(antenna2)
        # temp_C.append(time_ans_2)
        # temp_C.append(K_2)
        # A_process=np.hstack([temp_B,temp_C])
        # A_process.append(time_ans)
        # A_process.append(K)
        # A_process.append(time_ans_2)
        # A_process.append(K_2)
        # plt.suptitle('final')
        # plt.subplot(211)
        # plt.plot(K,linewidth=0.5)
        # plt.subplot(212)
        # plt.plot(K_2,linewidth=0.5)
        # plt.show()
        # pd.DataFrame(data1).round(5).to_csv('./1_lttb.csv', header=False, index=False)
        # pd.DataFrame(data2).round(5).to_csv('./2_lttb.csv', header=False, index=False)
        # return data1,data2
        return A_process

    def dataProcess_nomean(self,A):
        A=A.to_numpy()
        A[:,3]= A[:,3]*(2*math.pi/4096)
        phase=[]
        phase_ref=[]
        rss=[]
        rss_ref=[]
        time = []
        time_ref = []
        rss_2 = []
        rss_ref_2 = []
        phase_2 = []
        phase_ref_2 = []
        time_2 = []
        time_ref_2 = []
        # A_process=[]
        # A_process=np.ones((2,500,2))
        A_process = np.ones((2,500))
        temp_B=[]
        temp_C=[]
        for i in range(len(A)):
            if(A[i,0]==1107):
                if(A[i,1]==1):
                    phase.append(A[i,3])
                    time.append(A[i,6])
                    rss.append(A[i,4])
                elif(A[i,1]==2):
                    phase_2.append(A[i,3])
                    time_2.append(A[i,6])
                    rss_2.append(A[i,4])
            elif (A[i,0] == 1208):
                if (A[i,1] == 1):
                    phase_ref.append(A[i,3])
                    time_ref.append(A[i,6])
                    rss_ref.append(A[i,4])
                elif (A[i,1]== 2):
                    phase_ref_2.append(A[i,3])
                    time_ref_2.append(A[i,6])
                    rss_ref_2.append(A[i,4])
        for i in range(len(phase)-1):
            while phase[i+1]-phase[i]>2:
                phase[i+1]=phase[i+1]-math.pi
            while phase[i]-phase[i+1]>2:
                phase[i+1]=phase[i+1]+math.pi
        for i in range(len(phase_2)-1):
            while phase_2[i+1]-phase_2[i]>2:
                phase_2[i+1]=phase_2[i+1]-math.pi
            while phase_2[i]-phase_2[i+1]>2:
                phase_2[i+1]=phase_2[i+1]+math.pi
        for i in range(len(phase_ref)-1):
            while phase_ref[i+1]-phase_ref[i]>2:
                phase_ref[i+1]=phase_ref[i+1]-math.pi
            while phase_ref[i]-phase_ref[i+1]>2:
                phase_ref[i+1]=phase_ref[i+1]+math.pi
        for i in range(len(phase_ref_2)-1):
            while phase_ref_2[i+1]-phase_ref_2[i]>2:
                phase_ref_2[i+1]=phase_ref_2[i+1]-math.pi
            while phase_ref_2[i]-phase_ref_2[i+1]>2:
                phase_ref_2[i+1]=phase_ref_2[i+1]+math.pi
        # plt.suptitle('orignal')
        # plt.subplot(211)
        # x1=(time - time[0])/ 1000000
        # plt.plot(x1,phase, linewidth=0.5)
        # plt.subplot(212)
        # x2=(time_2-time_2[0])/1000000
        # plt.plot(x2,phase_2, linewidth=0.5)
        # plt.show()
        # phase=self.calculateKalmanFilter(phase)
        # phase_2=self.calculateKalmanFilter(phase_2)
        # phase_ref=self.calculateKalmanFilter(phase_ref)
        # phase_ref_2=self.calculateKalmanFilter(phase_ref_2)
        # phase = signal.medfilt(phase,21)
        # phase_2 = signal.medfilt(phase_2,21)
        # phase_ref = signal.medfilt(phase_ref,21)
        # phase_ref_2 = signal.medfilt(phase_ref_2,21)
        # plt.suptitle('filter')
        # plt.subplot(211)
        # x1 = (time - time[0]) / 1000000
        # plt.plot(x1, phase, linewidth=0.5)
        # plt.subplot(212)
        # x2 = (time_ref - time_ref[0]) / 1000000
        # plt.plot(x2, phase_2, linewidth=0.5)
        # plt.show()
        time1,r1=np.unique(time,return_index=True)
        for i in range(len(phase)-1,-1,-1):
            if ~np.isin(i,r1):
                phase=np.delete(phase,i)
        time2,r2= np.unique(time_ref,return_index=True)
        for i in range(len(phase_ref)-1,-1,-1):
            if ~np.isin(i, r2):
                phase_ref = np.delete(phase_ref, i)
        time1_2,r1_2= np.unique(time_2,return_index=True)
        for i in range(len(phase_2)-1,-1,-1):
            if ~np.isin(i, r1_2):
                phase_2 = np.delete(phase_2, i)
        time2_2,r2_2= np.unique(time_ref_2,return_index=True)
        for i in range(len(phase_ref_2)-1,-1,-1):
            if ~np.isin(i, r2_2):
                phase_ref_2 = np.delete(phase_ref_2, i)
        time_ans=np.sort(np.unique(np.hstack([time1,time2])))
        phase_yx=scipy.interpolate.PchipInterpolator(time1,phase,extrapolate=0)
        phase=phase_yx(time_ans)
        phase_ref_yx=scipy.interpolate.PchipInterpolator(time2,phase_ref,extrapolate=0)
        phase_ref=phase_ref_yx(time_ans)
        time_ans_2=np.sort(np.unique(np.hstack([time1_2,time2_2])))
        phase_2_yx=scipy.interpolate.PchipInterpolator(time1_2,phase_2,extrapolate=0)
        phase_2=phase_2_yx(time_ans_2)
        phase_ref_2_yx=scipy.interpolate.PchipInterpolator(time2_2,phase_ref_2,extrapolate=0)
        phase_ref_2=phase_ref_2_yx(time_ans_2)
        time_ans=time_ans-time_ans[0]
        s1=max(time1[0],time2[0])
        index_1=np.where(time_ans==s1)
        phase=np.delete(phase,index_1)
        phase_ref=np.delete(phase_ref,index_1)
        time_ans=np.delete(time_ans,index_1)
        time_ans=time_ans/1000000
        time_ans_2=time_ans_2-time_ans_2[0]
        s2=max(time1_2[0],time2_2[0])
        index_2=np.where(time_ans_2==s2)
        phase=np.delete(phase,index_2)
        phase_ref=np.delete(phase_ref,index_2)
        time_ans_2=np.delete(time_ans_2,index_2)
        time_ans_2=time_ans_2/1000000
        # for i in range(len(index_1)):
        K=phase-phase_ref
        K_2=phase_2-phase_ref_2
        for i in range(len(K)-1):
            while(K[i+1]-K[i]>2):
                K[i+1]=K[i+1]-math.pi
            while(K[i]-K[i+1]>2):
                K[i+1]=K[i+1]+math.pi
        for i in range(len(K_2) - 1):
            while (K_2[i + 1] - K_2[i] > 2):
                K_2[i + 1] = K_2[i + 1] - math.pi
            while (K_2[i] - K_2[i + 1] > 2):
                K_2[i + 1] = K_2[i + 1] + math.pi
        time_ans=time_ans[4:len(time_ans)-5]
        K=K[4:len(K)-5]
        time_ans_2=time_ans_2[4:len(time_ans_2)-5]
        K_2=K_2[4:len(K_2)-5]
        time_x=np.sort(np.unique(np.hstack([time_ans,time_ans_2])))
        ##去除nan值
        if np.isnan(np.min(K)):
            bool_nan_1=np.isnan(K)
            index_nan_1=np.where(bool_nan_1)
            a=index_nan_1[0]
            K=np.delete(K,a)
            time_ans=np.delete(time_ans,a)
            # for i in range(len(index_nan_1)-1,-1,-1):
            #     np.delete(K,index_nan_1[i])
            #     np.delete(time_ans,index_nan_1[i])
        if np.isnan(np.min(K_2)):
            bool_nan_2=np.isnan(K_2)
            index_nan_2=np.where(bool_nan_2)
            b=index_nan_2[0]
            K_2=np.delete(K_2,b)
            time_ans_2=np.delete(time_ans_2,b)
        # phase_fy1=scipy.interpolate.PchipInterpolator(time_ans,K,extrapolate=0)
        # phase_fy2=scipy.interpolate.PchipInterpolator(time_ans_2,K_2,extrapolate=0)
        # phase_y1=phase_fy1(time_x)
        # phase_y2=phase_fy2(time_x)
        # # phase_y=phase_y1-phase_y2
        # # A_process=phase_y
        # if np.isnan(np.min(phase_y1)):
        #     bool_nan=np.isnan(phase_y1)
        #     index_nan=np.where(bool_nan)
        #     b=index_nan[0]
        #     phase_y1=np.delete(phase_y1,b)
        # pad_value_1=phase_y1[len(phase_y1)-1]
        # phase_y1=np.pad(phase_y1,pad_width=(0,14800-phase_y1.shape[0]),mode='constant',constant_values=(0,pad_value_1))
        # if np.isnan(np.min(phase_y2)):
        #     bool_nan=np.isnan(phase_y2)
        #     index_nan=np.where(bool_nan)
        #     b=index_nan[0]
        #     phase_y2=np.delete(phase_y2,b)
        # pad_value_2=phase_y2[len(phase_y2)-1]
        # phase_y2=np.pad(phase_y2,pad_width=(0,14800-phase_y2.shape[0]),mode='constant',constant_values=(0,pad_value_2))
        # phase_y2=phase_y2-(phase_y2[0]-phase_y1[len(phase_y1-1)])
        phase_y1 =K
        # phase_y2 = K_2
        phase_y2 =K_2-(K_2[0]-K[len(K)-1])
        time_ans=time_ans[:,np.newaxis]
        time_ans_2=time_ans_2[:,np.newaxis]
        phase_y1=phase_y1[:,np.newaxis]
        phase_y2=phase_y2[:,np.newaxis]
        '''
        stft变化
          # aa = []
        # for i in range(200):
        #     aa.append(np.sin(0.3 * np.pi * i))
        # for i in range(200):
        #     aa.append(np.sin(0.13 * np.pi * i))
        # for i in range(200):
        #     aa.append(np.sin(0.05 * np.pi * i))
        # fs=2
        # f,t,nd=signal.stft(phase_y1,fs=fs,window='hann',nperseg=150,noverlap=0)
        # plt.pcolormesh(t,f,np.abs(nd), vmin = 0, vmax = 4)
        # plt.title('STFT')
        # plt.ylabel('frequency')
        # plt.xlabel('time')
        # plt.show()
        '''

        data1=np.hstack([time_ans,phase_y1])
        # pd.DataFrame(data1).round(5).to_csv('./1_lttb_old.csv', header=False, index=False)
        data2=np.hstack([time_ans_2,phase_y2])
        data1=lttb.downsample(data1,n_out=500)
        data2=lttb.downsample(data2,n_out=500)
        # plt.plot(time_ans_2,phase_y2)
        # plt.plot(data2[:, 0],data2[:, 1])
        # data2_normal=self.Zscore(data2[:, 1])
        data2_normal = self.normalize(data2[:, 1])
        # data2_normal = self.Zscore(data2_normal)
        # plt.plot(data2[:, 0],data2_normal)

        # plt.plot(time_ans,phase_y1)
        # plt.plot(data1[:, 0],data1[:, 1])
        # data1_normal=self.Zscore(data1[:, 1])
        data1_normal = self.normalize(data1[:, 1])
        # data1_normal = self.Zscore(data1_normal)
        # plt.plot(data1[:, 0],data1_normal)
        # plt.show()
        # phase_y1=torch.tensor(phase_y1)
        # phase_y2=torch.tensor(phase_y2)
        # phase_y1_rsa=F.interpolate(phase_y1,scale_factor=0.2)
        # phase_y2_rsa=F.interpolate(phase_y2,scale_factor=0.2)
        # A_process[0,:]=data1[:,1]
        # A_process[1, :] = data2[:, 1]
        # A_process[0,:,:]=data1[]
        # A_process=np.expand_dims(A_process,0)
        # A_process[1,:,:]=data2
        # data2_normal=data2_normal-(np.mean(data2_normal)-np.mean(data1_normal))
        # data2_normal = data2_normal-(np.mean(data1_normal)-data2_normal[0])
        A_process[0,:]=data1_normal
        A_process[1,:] = data2_normal

        # A_process = np.hstack([data1_normal,data2_normal])
        # if np.isnan(np.min(A_process)):
        #     bool_nan=np.isnan(A_process)
        #     index_nan=np.where(bool_nan)
        #     b=index_nan[0]
        #     A_process=np.delete(A_process,b)
        # A_process.append(phase_y1)
        # A_process.append(phase_y2)
        # A_process.extend(phase_y2)
        # pad_value=A_process[len(A_process)-1]
        # A_process = np.pad(A_process, pad_width=(0, 14800 - len(A_process)), mode='constant',
        #                   constant_values=(0, pad_value))
        # A_process = np.pad(A_process, pad_width=(0, 2960 - len(A_process)), mode='constant',
        #                  constant_values=(0, pad_value))
        # A_process=np.array(A_process)
        # phase_y=self.calculateKalmanFilter(phase_y)
        # plt.plot(data1_normal)
        # plt.plot(data2_normal)
        # plt.plot(A_process)
        # plt.plot(A_process[1])
        # plt.show()
        # antenna1=np.ones(len(time_ans))
        # antenna2=np.ones(len(time_ans_2))*2
        # temp_B.append(antenna1[0:])
        # temp_B.append(time_ans)
        # temp_B.append(K)
        # temp_C.append(antenna2)
        # temp_C.append(time_ans_2)
        # temp_C.append(K_2)
        # A_process=np.hstack([temp_B,temp_C])
        # A_process.append(time_ans)
        # A_process.append(K)
        # A_process.append(time_ans_2)
        # A_process.append(K_2)
        # plt.suptitle('final')
        # plt.subplot(211)
        # plt.plot(K,linewidth=0.5)
        # plt.subplot(212)
        # plt.plot(K_2,linewidth=0.5)
        # plt.show()
        # pd.DataFrame(data1).round(5).to_csv('./1_lttb.csv', header=False, index=False)
        # pd.DataFrame(data2).round(5).to_csv('./2_lttb.csv', header=False, index=False)
        # return data1,data2
        return A_process
    def calculateKalmanFilter(self,input_args):
        prevData = 0.0
        p = 10
        q = 0.001
        r = 0.08
        kGain = 0
        outData = []
        output_args=[]
        for i in range(len(input_args)):
            p=p+q
            kGain=p/(p+r)
            kGain = p / (p + r)
            temp = input_args[i]
            temp = prevData + (kGain * (temp - prevData))
            p = (1 - kGain) * p
            prevData = temp
            output_args.append(temp)
        return output_args

    def Zscore(self,data):
        x_mean = np.mean(data)
        length = len(data)
        vari = np.sqrt((np.sum((data - x_mean) ** 2)) / length)
    #     print('方差:', vari)
        data= (data- x_mean) / vari
    #     print('Z-score标准化后的矩阵是', data)
        return data
    # def sigmoid(self,data):
    #     return 1.0/(1+np.exp(-float(data)))
    def normalize(self,data):
        data=(data-np.min(data)) /(np.max(data)-np.min(data))
        return data

    def mean_downsample(signal, factor):
        """ 对时间序列进行均值降采样 """  # 确定每个样本点的宽度
        width = signal.shape[0] // factor # 计算每个样本点的平均值
        downsampled = np.mean(signal[:width*factor].reshape(-1, factor), axis=1)
        return downsampled
class mySubset(Subset):
    def __init__(self,dataset,indices):
        super().__init__(dataset,indices)

        target=[]
        for i in range(len(indices)):
            target.append(dataset.labels[indices[i]])
        # self.classes=dataset.classes
        self.targets=torch.tensor(target)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        x,y=self.dataset[self.indices[item]]
        return x,y

def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, test_scale=0.2):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
    :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train','test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        # val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        # val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            # elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            #     copy2(src_img_path, val_folder)
            #     # print("{}复制到了{}".format(src_img_path, val_folder))
            #     val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        # print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))

def savedata(target,data,labelname,dataname):
    data=np.array(data)
    split_path=os.path.join(target,labelname)
    if os.path.isdir(split_path):
        pass
    else:
        os.mkdir(split_path)
    path=os.path.join(target,labelname,dataname)
    pd.DataFrame(data).round(5).to_csv(path,header=False,index=False)




# my_dataset = torchvision.datasets.DatasetFolder('data/')
if __name__ == '__main__':
    # my_dataset = torchvision.datasets.DatasetFolder('E:/a毕设/网络/data/',transform=None,target_transform=None,extensions="csv",loader=lambda x:pd.read_csv(x))
    label_dir="../环境/PNLOS/"
    target="../process_pnlos/test"
    train_dataset=myDataSet(label_dir=label_dir)
    # data,_=train_dataset.__getitem__(1)
    # data=data.view(2,500)
    # data=np.array(data)
    # plt.plot(data[0,:])
    # plt.plot(data[1,:])
    # plt.show()
    # data_x=data.permute(1,0)
    # data=np.array(data)
    # data_x=np.array(data_x)
    for i in range(len(train_dataset)):
        data,_,labelname,dataname=train_dataset.__getitem__(i)
        savedata(target,data,labelname,dataname)
    # src_data="../data/"
    # target_data="../meta_data/"
    # data_set_split(src_data,target_data)
    # A=np.array([1,2,3,4,5,6,7,8])
    # B=np.array([2,3,5])
    # A=np.delete(A,B)
    # print(A)
    print('123')