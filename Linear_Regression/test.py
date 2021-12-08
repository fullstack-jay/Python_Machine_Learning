import numpy as np
import pandas as pd
import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1)) # menyusun features
y = np.array(data[predict]) # menyusun labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.05)

best = 0
'''
for _ in range(50): # training dengan iterasi
    # membagi data train dan testing untuk features dan labels
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.05)
    linear = linear_model.LinearRegression() # menggunakan algoritma Linear Regression
    linear.fit(x_train, y_train) # melakukan training
    acc = linear.score(x_test, y_test) # mendapatkan akurasi hasil training
    print(acc)
    if acc > best: # mencari best accuracy dan di-save ke file pickle
        with open("student_model.pickle", "wb") as file:
            pickle.dump(linear, file)'''

pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in) # membuka file pickle

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test) # memprediksi G3
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) # membandingkan hasil prediksi dengan output aslinya

# mencaria korelasi antara G3 dengan feature lain dalam bentuk diagram matplotlib
p = "studytime"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()