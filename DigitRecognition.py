__author__ = 'pranavagarwal'
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import cross_validation

data = pd.read_csv("/Users/pranavagarwal/Documents/DigitalRecogniser/train.csv")
data = data.values
data = data[:100]
n_samples = data.shape[0]

# Separating the labels from the training set
train = data[:, 1:]
labels = data[:, :1]
knn = KNeighborsClassifier(weights = 'distance', n_neighbors=10, p=3)
# The learning is done on the first half of the dataset

knn.fit(train[:n_samples / 2], labels[:n_samples / 2])
expected = labels[n_samples / 2:]
predicted = knn.predict(train[n_samples / 2:])
def plot_confusion_matrix(y_pred, y):
    plt.imshow(metrics.confusion_matrix(y, y_pred),
               cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.xlabel('true value')
    plt.ylabel('predicted value')
# print("Classification report for classifier %s:\n%s\n" % (knn, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print "classification accuracy:", metrics.accuracy_score(expected, predicted)
plot_confusion_matrix(predicted, predicted)
plt.show()
'''
scores = cross_validation.cross_val_score(knn, train, labels, cv=3)
print(scores.mean())
'''
