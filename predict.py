#predict.py

import numpy as np
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.preprocessing import StandardScaler
import cPickle as pickle

model = pickle.load(open('data/powell_model.pkl', 'rb'))

df_test = pd.read_csv('data/test_script_examples.csv')

def falsy(df):
    #Count number of 'falsy' or suspicious zero/missing/nan type entries. Sum these and add extra column with count for each row.
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

def clean_data(df):
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

    cols_to_exclude = ['venue_state', 'venue_name', 'venue_country', 'venue_address', 'ticket_types', 'email_domain', 'description', 'previous_payouts', 'payee_name', 'org_name', 'org_desc', 'object_id', 'name', 'acct_type', 'country', 'listed', 'currency', 'payout_type', 'channels']

    df['const'] = 1

    X = df.drop(cols_to_exclude + ['fraud'], axis=1).values
    y = df['fraud'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def test_one():
    rand = np.random.randint(0, df_test.shape[0])
    df_clean = clean_data(df_test)
    test = df_clean.iloc[rand, :]
    y_pred = model.predict(test)
    print y_pred

test_one()