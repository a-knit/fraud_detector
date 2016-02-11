#app.py

import numpy as np
import pandas as pd
import json
import math
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
import requests
from flask import Flask, url_for,request, session, render_template
import socket
from pymongo import MongoClient

my_ip = socket.gethostbyname(socket.gethostname())
#my_ip = 10.3.1.97
predictions = []
app = Flask(__name__)

db_client = MongoClient()
db = db_client['fraud']
table = db['events']

@app.route('/score', methods=['POST'])
def api_score():
    re = request.json
    X = build_feat_mats(re)
    model = pickle.load(open('data/powell_model.pkl', 'rb'))
    y_pred = model.predict(X)
    predictions.append(y_pred[0])
    info = {}
    info['event'] = re
    info['prediction'] = y_pred[0]
    table.insert(info)
    return 'data added'


@app.route('/')
def api_root():
    statement = 'Predicted Probabilities of Fraud:' + str(predictions)
    return statement

@app.route('/dashboard')
def api_dashboard():
    rand_num = np.random.randint(220)
    data = table.find().skip(rand_num).limit(1).next()
    text = 'The most recent transaction has the following probability of fraud: '
    pred = data['prediction']
    pred_str = '%.2f%%' % (pred * 100)
    if pred < 0.33:
        pic = 'img/fonz_low.jpg'
        level = 'Low'
        button = 'btn btn-success'
    elif pred < 0.67:
        pic = 'img/slippery.jpg'
        level = 'Medium'
        button = 'btn btn-warning'
    else:
        pic = 'img/defcon1.jpg'
        level = 'High'
        button = 'btn btn-danger'
    

    return render_template('index.html', text=text, pred=pred_str, pic=pic, level=level, button=button)


    # '''
    # <body>
    # <h1>Fraud Dashboard </h1>
    # %s <b>%s</b>
    # <br>
    # <img src={{ url_for('static', filename = %s) }}>
    # </body>
    # ''' % (text, pred_str, pic)

def register(my_ip):
    reg_url = 'http://10.3.34.86:5000/register'
    print 'attempting to register {}'.format(my_ip)
    requests.post(reg_url, data={'ip': my_ip, 'port': 7777})
    print 'registered'

def build_feat_mats(re):
    columns = [u'approx_payout_date',        u'body_length',    u'delivery_method',
            u'event_created',          u'event_end',    u'event_published',
              u'event_start',       u'fb_published',                u'gts',
            u'has_analytics',         u'has_header',           u'has_logo',
              u'name_length',          u'num_order',        u'num_payouts',
             u'org_facebook',        u'org_twitter',      u'sale_duration',
           u'sale_duration2',           u'show_map',           u'user_age',
             u'user_created',          u'user_type',     u'venue_latitude',
          u'venue_longitude',              u'falsy',         u'listed_num',
               u'no_paytype',              u'CHECK',              u'const']
    X = np.zeros(len(columns))
    standard_columns = [u'approx_payout_date', u'body_length', u'delivery_method', u'event_created', u'event_end', u'event_published', u'event_start', u'fb_published', u'gts', u'has_analytics', u'has_header', u'has_logo', u'name_length', u'num_order', u'num_payouts', u'org_facebook', u'org_twitter', u'sale_duration', u'sale_duration2', u'show_map', u'user_age', u'user_created', u'user_type', u'venue_latitude', u'venue_longitude']
    columns_to_engineer = [u'falsy', u'listed_num', u'no_paytype', u'CHECK', u'const']
    for i, col in enumerate(columns):
        if col in standard_columns:
            if re[col] != re[col]:
                X[i] = 0
            else:
                X[i] = re[col]
            continue
        if col == 'falsy':
            X[i] = falsy(re)
            continue
        if col == 'listed_num':
            if re['listed'] == 'y':
                X[i] = 1
            continue
        if col == 'no_paytype':
            if re['payout_type'] == '':
                X[i] = 1
        if col == 'CHECK':
            if re['payout_type'] == 'CHECK':
                X[i] = 1
        if col == 'const':
            X[i] = 1

    scaler_model = pickle.load(open('data/scaler.pkl', 'rb'))
    X = scaler_model.transform(X)
    return X

def falsy(re):
    count = 0
    for key, value in re.iteritems():
        if key == 'fraud':
            continue
        if not value:
            count += 1
            continue
        if value != value:
            count += 1
            continue
        if value == 'n':
            count += 1
    return count


if __name__ == '__main__':
    register(my_ip)

    app.run('0.0.0.0',port=7777,debug=True)
    