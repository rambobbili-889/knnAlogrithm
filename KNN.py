import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report

class KNN:
    def __init__(self, location):
        try:
            self.df = pd.read_csv(location)
            self.df = self.df.drop(['id', 'Unnamed: 32'], axis=1)
            self.df['diagnosis'] = self.df['diagnosis'].map({'M': 0,'B': 1})
            self.X = self.df.iloc[:, 1:]
            self.Y = self.df.iloc[:, 0]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2,random_state=99)
            self.reg = KNeighborsClassifier(n_neighbors=7)
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def training_data(self):
        try:
            self.reg.fit(self.x_train,self.y_train)
            self.y_train_predict = self.reg.predict(self.x_train)

            print(f'The accuracy of the training data :{accuracy_score(self.y_train,self.y_train_predict)}')
            print(f'The classfication report for training data is :\n {classification_report(self.y_train, self.y_train_predict)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def testing_data(self):
        try:
            self.y_test_predict = self.reg.predict(self.x_test)
            print(f'The accuracy of the testing  data : {accuracy_score(self.y_test, self.y_test_predict)}')
            print(f'The classification report for testing  data is :\n{classification_report(self.y_test, self.y_test_predict)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def showing_testing(self):
        try:
            self.data_test = pd.DataFrame()
            self.data_test = self.x_test.copy()
            self.data_test['y_test_values'] = self.y_test
            self.data_test['y_test_predict'] = self.y_test_predict
            print(self.data_test)
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)


    def choosing_best_k_value(self):
        try:
            self.a = np.arange(1,101,2)
            self.acc=[]
            for i in self.a:
                self.reg = KNeighborsClassifier(n_neighbors=i)
                self.reg.fit(self.x_train,self.y_train)
                self.acc.append(self.reg.score(self.x_test,self.y_test))
            print(f'The best value of k is: {(self.a[self.acc.index(max(self.acc))])}')
            print(f'The highest accuracy is: {max(self.acc)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

if __name__ == "__main__":
    try:
        obj = KNN('C:\\Users\\Bunny\\PycharmProjects\\Machine_Learning_algorithms\\breast-cancer.csv')
        obj.training_data()
        obj.testing_data()
        obj.showing_testing()
        obj.choosing_best_k_value()
    except Exception as e:
        error_type, error_message, error_line_no = sys.exc_info()
        print(error_type, error_message, error_line_no.tb_lineno)