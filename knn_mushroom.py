##!/usr/bin/env python
## coding: utf-8
#

import pandas as pd
import numpy as np
from collections import Counter

# from sklearn.metrics import accuracy_score

# from google.colab import drive
# drive.mount("/content/drive")

class KNNClassifier:
    list_all = {}
    k = 1
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    def get_all_labels(self):
        self.list_all[0] = ['b','c','x','f','k','s']
        self.list_all[1] = ['f', 'g' , 'y', 's']  
        self.list_all[2] = ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']
        self.list_all[3] = ['t', 'f']
        self.list_all[4] = ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']
        self.list_all[5] = ['a', 'f', 'd', 'n']
        self.list_all[6] = ['c', 'w' , 'd']
        self.list_all[7] = ['b', 'n']
        self.list_all[8] = ['k', 'n' , 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y']
        self.list_all[9] = ['e', 't']
        self.list_all[10] = ['b','c','u','e','z','r']
        self.list_all[11] = ['f', 'y' , 'k', 's']
        self.list_all[12] = ['f', 'y' , 'k', 's']
        self.list_all[13] = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']
        self.list_all[14] = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']
        self.list_all[15] = ['p', 'u']
        self.list_all[16] = ['n', 'o', 'w', 'y']
        self.list_all[17] = ['n', 'o', 't']
        self.list_all[18] = ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z']
        self.list_all[19] = ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y']
        self.list_all[20] = ['a', 'c', 'n', 's', 'v', 'y']
        self.list_all[21] = ['g', 'l', 'm', 'p', 'u', 'w', 'd']
            

    def clean_train_data(self, df):
        col = df.iloc[:,11]
        df.iloc[:,11].replace(to_replace = "?", value = col.mode()[0], inplace=True)
        return df

    def clean_test_data(self, df):
        for col_name in df.columns:
            col=df[col_name]
            df[col_name].replace(to_replace = "?", value = col.mode()[0], inplace=True)
            return df
    
    def train_validation_split(self, df):
        dataLen=int(0.9*df.shape[0])
        # dataLen=df.shape[0]-60
        return df.iloc[0:dataLen, :], df.iloc[dataLen:,:]

    def re_idx(self, df_dum, col_idx):
        df_dum = df_dum.T.reindex(self.list_all[col_idx]).T.fillna(0)
        return df_dum
        
    def one_hot_encode(self, df):
#         print("one hot df:", df)
        self.get_all_labels()

        df_final = pd.DataFrame() #empty dataframe
        for col_idx in range(0, df.shape[1]):
            df_dum = pd.get_dummies(df.iloc[:,col_idx], columns = self.list_all[col_idx])
            df_dum = self.re_idx(df_dum, col_idx)
#             df_dum = df_dum.T.reindex(self.list_all[col_idx]).T.fillna(0)
            # print(col_idx, df_dum)
            df_final = pd.concat([df_final, df_dum], axis=1)
        return df_final
    
    def check_accuracy(self, predicted_label, validation_label):
        total=0
        correct=0
        accuracy=0.0
        if(len(predicted_label)!=len(validation_label)):
            print("Something's fishy!")
            return
            
        total=len(predicted_label)
        for i in range(total):
            if(predicted_label[i]==validation_label[i]):
                correct+=1
        accuracy=correct/total
        return accuracy
    
    def euclidean_distance(self, x, y):
        return np.linalg.norm(np.array(x)-np.array(y))
    
    def manhattan_distance(self, x, y):
        return np.sum(np.absolute(np.array(x)-np.array(y)))

    def knn_algo(self, train_data, test_data, k):
        predicted_label=[]
        for i in range(0, test_data.shape[0]): #test_data
            # print("i ",i)
            distance=[]
            labels=[]            
            for j in range(0, train_data.shape[0]): #train_data
                dist = self.euclidean_distance(test_data[i], train_data[j])
                #dist = self.manhattan_distance(test_data[i], train_data[j])
                distance.append([dist,label_of_train_data[j][0]])
            for j in sorted(distance)[:k]:
                labels.append(j[1])#gives the top k labels for a particular test_data
            predicted_label.append(Counter(labels).most_common(1)[0][0])
        return predicted_label

    def train_util(self,validation_df):
        self.train_df = self.one_hot_encode(self.train_df)    
        validation_df = self.one_hot_encode(validation_df) 
        return validation_df
    
    def predict_util(self):
        self.test_df = self.one_hot_encode(self.test_df) 
        return self.knn_algo(self.train_df.values, self.test_df.values, self.k)

    def predict(self, test_file_name):
        self.test_df = pd.read_csv(test_file_name, header=None)[:]
        self.test_df = self.clean_test_data(self.test_df)
        return self.predict_util()

    def train(self, train_file_name):
        df = pd.read_csv(train_file_name)[:]
        df = self.clean_train_data(df)

        self.train_df, validation_df = self.train_validation_split(df)

        global label_of_train_data
        label_of_train_data = self.train_df.iloc[:,:1].to_numpy()
        self.train_df = self.train_df.iloc[:,1:]#dropping the label col from train_data

        label_of_validation_data = validation_df.iloc[:, :1].to_numpy()
        validation_df = validation_df.iloc[:,1:]
        
        validation_df = self.train_util(validation_df)
        
        global accuracy_k
        accuracy_k = []
#         for i in [1,3,5,7]:
#             self.k=i
#             print("K:",self.k)
            
#             predicted_label = self.knn_algo(self.train_df.values, validation_df.values, self.k)
#             #print("predicted_label:", predicted_label)
#             accuracy = self.check_accuracy(predicted_label, label_of_validation_data)            
#             accuracy_k.append([accuracy, self.k])
        self.k=5       

# knn_classifier = KNNClassifier()
# knn_classifier.train('./Datasets/q2/train.csv')
# predictions = knn_classifier.predict('./Datasets/q2/test.csv')
# test_labels = list()
# with open("./Datasets/q2/test_labels.csv") as f:
#  for line in f:
#    test_labels.append(line.strip())
# # print("test_labels: ", test_labels)
# print (accuracy_score(test_labels[:], predictions))