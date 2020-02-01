#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from collections import Counter

# from google.colab import drive
# drive.mount("/content/drive")


class KNNClassifier:
    k = 1
    train_df = pd.DataFrame()

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
            # print("i:",i)
            distance=[]
            labels=[]            
            for j in range(0, train_data.shape[0]): #train_data
                dist = self.euclidean_distance(test_data[i], train_data[j])
#                 dist = self.manhattan_distance(test_data[i], train_data[j])
                distance.append([dist,label_of_train_data[j][0]])
                
            for j in sorted(distance)[:k]:
                labels.append(j[1])#gives the top k labels for a particular test_df
            predicted_label.append(Counter(labels).most_common(1)[0][0])
        return predicted_label


    def train_validation_split(self, df):
        dataLen=int(0.9*df.shape[0])
#         dataLen=df.shape[0]-10
        return df.iloc[0:dataLen, :], df.iloc[dataLen:,:]


    def predict(self, test_file_name):
        test_df = pd.read_csv(test_file_name, header=None)[:]
        return self.knn_algo(self.train_df.values, test_df.values, self.k)


    def train(self, train_file_name):
        df = pd.read_csv(train_file_name)[:]

        self.train_df, validation_df = self.train_validation_split(df)

        global label_of_train_data
        label_of_train_data = self.train_df.iloc[:,:1].to_numpy()
        self.train_df = self.train_df.iloc[:,1:]#dropping the label col from train_data

        label_of_validation_data = validation_df.iloc[:, :1].to_numpy()
        validation_df = validation_df.iloc[:,1:]

        self.k=3
        #predicted_label=self.knn_algo(self.train_df.values, validation_df.values, self.k)

#knn_classifier = KNNClassifier()
#knn_classifier.train('./Datasets/q1/train.csv')
#predictions = knn_classifier.predict('./Datasets/q1/test.csv')
#test_labels = list()
#with open("./Datasets/q1/test_labels.csv") as f:
#  for line in f:
#    test_labels.append(int(line))
#print(accuracy_score(test_labels[:10], predictions))


# In[ ]:




