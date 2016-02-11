#pair.py - Nash & Alex - Fraud Case Study

import numpy as np
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
import cPickle as pickle



#Count number of 'falsy' or suspicious zero/missing/nan type entries. Sum these and add extra column with count for each row.
def falsy(df):
	total = []
	for index, row in df.iterrows():
		count = 0
		for col in df.columns:
			if col == 'fraud':
				continue
			if col == 'falsy':
				continue
			entry = row[col]
			if not entry:
				count += 1
				continue
			if entry != entry:
				count += 1
				continue
			if entry == 'n':
				count += 1

		total.append(count)
	return pd.Series(total)

def clean_data(filename):
	df = pd.read_json(filename)
	df['falsy'] = falsy(df)

	#create type of fraud events. Create extra column of 1 for fraud event and 0 for all other events.
	fraud_events = ['fraudster_event', 'frauster', 'fraudster_att']
	df['fraud'] = 0
	df['fraud'][df['acct_type'].isin(fraud_events)] = 1
	df['listed_num'] = 0
	df['listed_num'][df['listed']=='y'] = 1

	#create dummie variables for 'payout_type' and make 'ACH' the baseline since it has the highest number of occurances
	df[['no_paytype', 'ACH', 'CHECK']] = pd.get_dummies(df['payout_type'])
	df = df.drop('ACH', axis=1)

	#fill NaN values with zero so we can run logistic regression
	df = df.fillna(value=0)
	return df

def smote(X, y, target, k=None):
	"""
	INPUT:
	X, y - your data
	target - the percentage of positive class 
			 observations in the output
	k - k in k nearest neighbors
	OUTPUT:
	X_oversampled, y_oversampled - oversampled data
	`smote` generates new observations from the positive (minority) class:
	For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
	"""
	if target <= sum(y)/float(len(y)):
		return X, y
	if k is None:
		k = len(X)**.5
	# fit kNN model
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X[y==1], y[y==1])
	neighbors = knn.kneighbors()[0]
	positive_observations = X[y==1]
	# determine how many new positive observations to generate
	positive_count = sum(y)
	negative_count = len(y) - positive_count
	target_positive_count = target*negative_count / (1. - target)
	target_positive_count = int(round(target_positive_count))
	number_of_new_observations = target_positive_count - positive_count
	# generate synthetic observations
	synthetic_observations = np.empty((0, X.shape[1]))
	while len(synthetic_observations) < number_of_new_observations:
		obs_index = np.random.randint(len(positive_observations))
		observation = positive_observations[obs_index]
		neighbor_index = np.random.choice(neighbors[obs_index])
		neighbor = X[neighbor_index]
		obs_weights = np.random.random(len(neighbor))
		neighbor_weights = 1 - obs_weights
		new_observation = obs_weights*observation + neighbor_weights*neighbor
		synthetic_observations = np.vstack((synthetic_observations, new_observation))

	X_smoted = np.vstack((X, synthetic_observations))
	y_smoted = np.concatenate((y, [1]*len(synthetic_observations)))

	return X_smoted, y_smoted

def prep_X_y(df, constant=False, split=True):
	cols_to_exclude = ['venue_state', 'venue_name', 'venue_country', 'venue_address', 'ticket_types', 'email_domain', 'description', 'previous_payouts', 'payee_name', 'org_name', 'org_desc', 'object_id', 'name', 'acct_type', 'country', 'listed', 'currency', 'payout_type', 'channels']

	if constant:
		df['const'] = 1

	X = df.drop(cols_to_exclude + ['fraud'], axis=1).values
	y = df['fraud'].values

	print 'columns used:\n', df.drop(cols_to_exclude + ['fraud'], axis=1).columns

	if split:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.fit_transform(X_test)

		X_smoted, y_smoted = smote(X_train, y_train, target=.5)
		return X_smoted, X_test, y_smoted, y_test
	else:
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
		X_smoted, y_smoted = smote(X, y, target=.5)
		return X_smoted, y_smoted


def random_forest():
	X_smoted, X_test, y_smoted, y_test = prep_X_y(df, constant=False)
	rf = RandomForestClassifier(max_features=5, max_depth=4)
	rf.fit(X_smoted, y_smoted)
	y_pred = rf.predict(X_test)
	print 'Random Forest--------------------------------'
	print 'Confusion Matrix:', confusion_matrix(y_test, y_pred)
	print 'Accuracy:', accuracy_score(y_test, y_pred)
	print 'Precision:', precision_score(y_test, y_pred)
	print 'Recall:', recall_score(y_test, y_pred)
	return rf

def logistic_regression():
	X_smoted, X_test, y_smoted, y_test = prep_X_y(df, constant=True)
	lm = LogisticRegression()
	lm.fit(X_smoted, y_smoted)
	y_pred = lm.predict(X_test).round(0)
	print 'Logistic Regression--------------------------------'
	print 'Confusion Matrix:', confusion_matrix(y_test, y_pred)
	print 'Accuracy:', accuracy_score(y_test, y_pred)
	print 'Precision:', precision_score(y_test, y_pred)
	print 'Recall:', recall_score(y_test, y_pred)
	return lm

def svm():
	X_smoted, X_test, y_smoted, y_test = prep_X_y(df, constant=False)
	svm = SVC(C=.001).fit(X_smoted, y_smoted)
	y_pred = svm.predict(X_test).round(0)
	print 'SVM--------------------------------'
	print 'Confusion Matrix:', confusion_matrix(y_test, y_pred)
	print 'Accuracy:', accuracy_score(y_test, y_pred)
	print 'Precision:', precision_score(y_test, y_pred)
	print 'Recall:', recall_score(y_test, y_pred)
	return svm

def logit_reg():
	X_smoted, X_test, y_smoted, y_test = prep_X_y(df, constant=True)
	lm = Logit(y_smoted, X_smoted).fit(method = 'powell')
	y_pred = lm.predict(X_test).round(0)
	print 'Statsmodels Logit Regression--------------------------------'
	print 'Confusion Matrix:', confusion_matrix(y_test, y_pred)
	print 'Accuracy:', accuracy_score(y_test, y_pred)
	print 'Precision:', precision_score(y_test, y_pred)
	print 'Recall:', recall_score(y_test, y_pred)
	return lm

def main():
	# random_forest()
	# logistic_regression()
	# svm()
	logit_reg()
	# pickle_logit()

def pickle_logit():
	X_smoted, y_smoted = prep_X_y(df, constant=True, split=False)
	lm = Logit(y_smoted, X_smoted).fit(method='powell')
	with open('data/powell_model.pkl', 'w') as f:
		pickle.dump(lm, f)

def make_test_set(df):
	n = df.shape[0]
	choices = np.random.randint(0, n, 4)
	choices = np.append(choices, [0])
	df_test = df.iloc[choices, :]
	df_test.to_csv('data/test_script_examples.csv')

def pickle_scaler(df):
	cols_to_exclude = ['venue_state', 'venue_name', 'venue_country', 'venue_address', 'ticket_types', 'email_domain', 'description', 'previous_payouts', 'payee_name', 'org_name', 'org_desc', 'object_id', 'name', 'acct_type', 'country', 'listed', 'currency', 'payout_type', 'channels']
	df['const'] = 1

	X = df.drop(cols_to_exclude + ['fraud'], axis=1).values
	y = df['fraud'].values

	scaler = StandardScaler()
	mod = scaler.fit(X)
	with open('data/scaler.pkl', 'w') as f:
		pickle.dump(mod, f)

if __name__ == '__main__':
	data = 'data/train_new.json'
	# df_orig = pd.read_json(data)
	df = clean_data(data)
	pickle_scaler(df)
	# main()
