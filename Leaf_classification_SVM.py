import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pylab as pl
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


warnings.filterwarnings('always')

leaf = pd.read_csv('leaf.csv')
leaf.head()

print(leaf.shape)

print(leaf['Specimen'].unique())

print(leaf.groupby('Specimen').size())

sns.countplot(leaf['Specimen'],label="Count")
plt.savefig('count_leaf')
plt.show()

leaf.drop('Class' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('leaf_hist')
plt.show()

feature_names =["Eccentricity" ,"Elongation","Solidity","Stochastic_convexity","Lobedness","Avd_intensity","Avg_contrast","smoothness","Third_moment","Uniformity",	"Entropy"]
X = leaf[feature_names]
y = leaf['Class']
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('leaf_scatter_matrix')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))


pred = svm.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))














