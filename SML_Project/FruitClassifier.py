import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Load the training and testing data
label_map = {
    'Apple_Raw': 0,
    'Apple_Ripe': 1,
    'Banana_Raw': 2,
    'Banana_Ripe': 3,
    'Coconut_Raw': 4,
    'Coconut_Ripe': 5,
    'Guava_Raw': 6,
    'Guava_Ripe': 7,
    'Leeche_Raw': 8,
    'Leeche_Ripe': 9,
    'Mango_Raw': 10,
    'Mango_Ripe': 11,
    'Orange_Raw': 12,
    'Orange_Ripe': 13,
    'Papaya_Raw': 14,
    'Papaya_Ripe': 15,
    'Pomengranate_Raw': 16,
    'Pomengranate_Ripe': 17,
    'Strawberry_Raw': 18,
    'Strawberry_Ripe': 19
}

train_data['category'] = train_data['category'].map(label_map)
train_data['category'] = train_data['category'].astype('int')

data = pd.read_csv('train.csv')
data1 = pd.read_csv('test.csv')
X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, 1:]

X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

pca = PCA(n_components=400)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

lda = LDA(n_components=19)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# for i in range(2 , 21):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(X_train)
#     silhouette_scores = silhouette_score(X_train, kmeans.labels_)
#     print("Mean Silhouette score:",i, silhouette_scores)

kmeans = KMeans(n_clusters = 20)
kmeans.fit(X_train)
clusters = kmeans.predict(X_train)
X_train=np.column_stack([X_train,clusters])

clusters = kmeans.predict(X_test)
X_test=np.column_stack([X_test,clusters])

lof = LocalOutlierFactor(n_neighbors=20)
out = lof.fit_predict(X_train)
c=0
for i in out:
    if i==-1:
        c+=1
        
#print(c)

nn1 = MLPClassifier(hidden_layer_sizes=1000, 
                   activation='relu', 
                   solver='adam', 
                   alpha=0.001, 
                   learning_rate='constant', 
                   learning_rate_init=0.01, 
                   max_iter=3000, 
                   batch_size=32, 
                   verbose=True)

nn2 = MLPClassifier(hidden_layer_sizes=1000, 
                   activation='logistic', 
                   solver='adam', 
                   alpha=0.001, 
                   learning_rate='constant', 
                   learning_rate_init=0.01, 
                   max_iter=3000, 
                   batch_size=32, 
                   verbose=True)

lr1 = LogisticRegression(C = 0.1, max_iter= 50000, penalty= 'l2', solver ='newton-cg')
lr2 = LogisticRegression(C = 0.1, max_iter= 50000, penalty= 'l2', solver ='liblinear')

voting_clf = VotingClassifier(estimators=[('lr1',lr1),('lr2',lr2),('nn1',nn1),('nn2',nn2)], voting='hard')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
inv_label_map = {v: k for k, v in label_map.items()}
# Creating a new list with inverted labels for y_pred
inv_y_pred = [inv_label_map[pred] for pred in y_pred]
y_pred = pd.DataFrame(inv_y_pred)
y_pred.columns = ['Category']
y_pred.to_csv('output.csv', index=True)

kf=KFold(n_splits=10)
scores = []
xtrain = []
xtest = []
ytrain = []
ytest = []
for train_index, test_index in kf.split(X_train):
    for i in train_index:
        xtrain.append(X_train[i])
        ytrain.append(y_train[i])
    for i in test_index:
        xtest.append(X_train[i])
        ytest.append(y_train[i])
    voting_clf.fit(xtrain, ytrain)
    y_pred = voting_clf.predict(xtest)
    score = accuracy_score(ytest, y_pred)
    scores.append(score)

print(scores)
