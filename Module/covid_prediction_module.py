# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:05:00 2022

@author: _K
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout


class EDA():
    def __init__(self):
        pass

    def plot_graph(self,df,column_names):
        
        for col in column_names:
            plt.figure()
            plt.plot(df[col])
            plt.legend(col)
            plt.show()


class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self,X_train,num_node=64,
                          drop_rate1=0.3,drop_rate2=0.2,output_node=1):
        model=Sequential()
        model.add(Input(shape=(np.shape(X_train)[1],1))) #input_length #features 
        model.add(LSTM(num_node,return_sequences=True)) # only once return_se=true when LSTM meet LSTM after
        model.add(Dropout(drop_rate2))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate2))
        model.add(Dense(128)) # Hidden layer 2
        model.add(Dropout(drop_rate2))
        model.add(Dense(64)) # Hidden layer 2
        model.add(Dropout(drop_rate2))
        model.add(Dense(output_node,activation='linear')) #output layer
        model.summary()
        
        return model



class model_evaluation():
    def plot_predicted_graph(self,test_df,predicted,mms):
        plt.figure()
        plt.plot(test_df,'b',label='actual covid-19 cases')
        plt.plot(predicted,'r',label='predicted covid-19 cases')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='actual covid-19 cases')
        plt.plot(mms.inverse_transform(predicted),'r',label='predicted covid-19 cases')
        plt.legend()
        plt.show()



















