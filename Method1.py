
# coding: utf-8

# In[62]:

import pandas as pd

from sklearn.cross_validation import KFold
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier as DT


training = pd.read_csv('Training_Dataset.csv')


def clean_data(training, isTrain):
    training['mvar33_refined'] = 0
    for i,item in enumerate(training['mvar33']):
        training.ix[i,'mvar33_refined'] = (ord(item[0]) - ord('A'))*676 + (ord(item[1])-ord('A'))*26 + (ord(item[2]) - ord('A'))
    training['mvar32_refined'] = 0
    for i,item in enumerate(training['mvar32']):
        training.ix[i,'mvar32_refined'] = (ord(item[0]) - ord('A'))*676 + (ord(item[1])-ord('A'))*26 + (ord(item[2]) - ord('A'))
    age_dict = sorted(training['mvar27'].unique())
    age_dict_mapping = dict(zip(age_dict, 
                                     range(0, len(age_dict) + 1)))
    training['mvar27_val'] = training['mvar27'] \
                                   .map(age_dict_mapping) \
                                   .astype(int)
    training['mvar30'] = training['mvar30'].fillna('nan')
    education_dict = sorted(training['mvar30'].unique())
    education_dict_mapping = dict(zip(education_dict, 
                                     range(0, len(education_dict) + 1)))
    training['mvar30_val'] = training['mvar30'] \
                                   .map(education_dict_mapping) \
                                   .astype(int)
    party_dict = sorted(training['party_voted_past'].unique())

    party_dict_mapping = dict(zip(party_dict, 
                                     range(0, len(party_dict) + 1)))
    training['party_voted_past_val'] = training['party_voted_past'] \
                                   .map(party_dict_mapping) \
                                   .astype(int)
    if (isTrain):
        training['actual_vote_val'] = training['actual_vote'] \
                                       .map(party_dict_mapping) \
                                        .astype(int)
        training['y'] = training['actual_vote_val']
    if (not isTrain):
        training = training.drop(['mvar33', 'mvar32', 'mvar30', 'mvar27', 'party_voted_past' ], axis = 1)
    else:
        training = training.drop(['actual_vote_val', 'actual_vote', 'mvar33', 'mvar32', 'mvar30', 'mvar27', 'party_voted_past', 'citizen_id' ], axis = 1)
    training['mvar28'] = training['mvar28'].fillna(2)
    training['mvar28'] = training['mvar28'].apply(int)
    training['mvar29'] = training['mvar29'].fillna(2)
    training['mvar29'] = training['mvar29'].apply(int)
    return training

training = clean_data(training, True)

y = training['y']

X = training.drop(['y'], axis = 1).as_matrix()


def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)



# print ("Decision Tree:")
# print ("%.3f" % accuracy(y, run_cv(X,y,DT)))

# print ("Logistic Regression:")
# print ("%.3f" % accuracy(y, run_cv(X,y,LR)))
# print ("Gradient Boosting Classifier")
# print ("%.3f" % accuracy(y, run_cv(X,y,GBC)))
# # print ("Support vector machines:")
# # print ("%.3f" % accuracy(y, run_cv(X,y,SVC)))
# print ("Random forest:")
# print ("%.3f" % accuracy(y, run_cv(X,y,RF)))
# print ("K-nearest-neighbors:")
# print ("%.3f" % accuracy(y, run_cv(X,y,KNN)))


# In[ ]:

############ Method 1
# clf = GBC()
# clf.fit(X,y)
# score = clf.score(X, y)
# print ("Mean accuracy of Random Forest: {0}".format(score))
# df_test = pd.read_csv('Leaderboard_Dataset.csv')
# df_test = clean_data(df_test, False)
# test_data = df_test.values
# test_x = test_data[:, 1:]
# y_pred = clf.predict(test_x)
# df_test['Voted_to'] = y_pred
# # {'Centaur': 0, 'Cosmos': 1, 'Ebony': 2, 'Odyssey': 3, 'Tokugawa': 4}
# rev_party_dict_mapping = {}
# rev_party_dict_mapping[0] = 'Centaur'
# rev_party_dict_mapping[1] = 'Cosmos'
# rev_party_dict_mapping[2] = 'Ebony'
# rev_party_dict_mapping[3] = 'Odyssey'
# rev_party_dict_mapping[4] = 'Tokugawa'
# df_test['Voted_to_ref'] = df_test['Voted_to'].map(rev_party_dict_mapping).astype(str)
# import csv
# df_test[['citizen_id', 'Voted_to_ref']] \
#     .to_csv('snark _IITKharagpur_1.csv', index = False, quoting=csv.QUOTE_NONNUMERIC, header = False)

############ Method 2
import tflearn
net = tflearn.input_data(shape=[None, 34])
net = tflearn.fully_connected(net, 68)
net = tflearn.fully_connected(net, 68, activation='sigmoid')
net = tflearn.fully_connected(net, 34)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 5, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
model = tflearn.DNN(net)

labels = np.zeros((len(y), 5))
for i in range(len(y)):
    labels[i][y[i]] = 1

model.fit(X, labels, n_epoch=10, batch_size=16, show_metric=True)

# df_test = pd.read_csv('Leaderboard_Dataset.csv')
# df_test = clean_data(df_test, False)
# test_data = df_test.values
# test_x = test_data[:, 1:]
# pred = model.predict(test_x)
# y_pred = []
# for i in range(len(pred)):
#     y_pred.append(pred[i].index(max(pred[i])))
# df_test['Voted_to'] = y_pred
# # {'Centaur': 0, 'Cosmos': 1, 'Ebony': 2, 'Odyssey': 3, 'Tokugawa': 4}
# rev_party_dict_mapping = {}
# rev_party_dict_mapping[0] = 'Centaur'
# rev_party_dict_mapping[1] = 'Cosmos'
# rev_party_dict_mapping[2] = 'Ebony'
# rev_party_dict_mapping[3] = 'Odyssey'
# rev_party_dict_mapping[4] = 'Tokugawa'
# df_test['Voted_to_ref'] = df_test['Voted_to'].map(rev_party_dict_mapping).astype(str)
# import csv
# df_test[['citizen_id', 'Voted_to_ref']] \
#     .to_csv('snark_IITKharagpur_5.csv', index = False, quoting=csv.QUOTE_NONNUMERIC, header = False)
