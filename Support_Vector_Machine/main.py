import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

classes = ['malignant' 'benign']

clf = svm.SVC(kernel="linear", C=2) # menggunakan SVC (bagian dari SVM)
# clf = KNeighborsClassifier(n_neighbors=9) # menggunakan KNN
clf.fit(x_train, y_train) # training

y_pred = clf.predict(x_test)

# membandingkan data testing dengan data hasil training
# sehingga menghasilkan nilai akurasi pelatihan
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)