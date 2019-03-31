# -*- coding: utf-8 -*-
"""
@author: 14307
"""

import numpy as np


class Linear_reg:
    
    
    def __init__(self, init_data):
        '''
        init_data数据格式为dataframe,最后一列为label
        '''
        
        input_data = init_data[init_data.columns[:-1]]
        input_data['e'] = [1]*len(input_data)
        self.input_data = input_data
        self.label = init_data[init_data.columns[-1]]
    
    
    def Sigmoid(self, x):
        
        return 1/(1+np.exp(-x))
 
    
    def Hessian(self, beta, length):
        '''
        计算梯度算子与Hessian矩阵
        '''
        
        input_data = self.input_data
        label = self.label
        gradient = np.mat([0]*length, dtype = np.float64).T
        hessian_mat = np.mat([[0]*length]*length, dtype = np.float64)
        
        for i in range(len(input_data)):
            row = np.mat(list(input_data.iloc[i])).T
            pro0 = float(self.Sigmoid(-beta*row))
            gradient += row*(1-pro0-label[i])
            hessian_mat += row * row.T *(1-pro0)*pro0
        return gradient, hessian_mat  
        
    
    def Newton(self, iter_num = 10):
        '''
        牛顿迭代法求解
        '''
        input_data = self.input_data
        length =len(input_data.iloc[0])
        beta_init = np.mat([0]*length)
        
        for i in range(iter_num):
            gradient, hessian = self.Hessian(beta_init, length)
            beta = beta_init - (hessian.I * gradient).T
            beta_init = beta
        return beta   
    
    
    def Predict(self, test_data, iter_num = 10):
        '''
        预测方法
        '''
        
        beta = self.Newton(iter_num)
        test_p = []
        for i in range(len(test_data)):       
            p =  float(np.mat(test_data.iloc[i])*beta.T)
            p =  self.Sigmoid(p)
            if p>=0.5:
                p = 1
            else: p = 0
            test_p.append(p)
        return test_p
    
    def Train_Pre(self):
        '''
        训练集结果以及正确率
        '''
        input_data = self.input_data
        label = self.label
        train_p = self.Predict(input_data)  
        T = 0
        for i, j in list(zip(train_p, label)):
            if i == j: T += 1
        true_rate = T/len(label)
        return train_p, true_rate
        

