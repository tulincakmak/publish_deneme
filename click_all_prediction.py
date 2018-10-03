# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:22:37 2018

@author: kerimtumkaya
"""

import smtplib
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats
import sqlalchemy
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from threading import Timer


def click_prediction_job():
    try:
        njobs = 32  # for running XGBoost
        logs_ctr = {'prediction_type': 'ctr'}
        logs_impr = {'prediction_type': 'impr'}

        """
        On-line training:
         xgb = XGBRegressor(learning_rate=0.1, max_depth=8, n_estimators=200, n_jobs=njobs) 
         xgb.fit(X, y, xgb_model=old_model_filename)
         """

        sql = "select * from pred_data_click_all_positions"

        # cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=otelz__1521458750;UID=caglaS;PWD=c:agla12S')
        # cursor = cnxn.cursor()
        # data = pd.read_sql(sql,cnxn)

        conn = 'mssql+pymssql://gizemaras:gzmrs@123@78.40.231.196/otelz__1521458750'
        engine = sqlalchemy.create_engine(conn, isolation_level="READ COMMITTED", echo=False, pool_size=50, max_overflow=0)
        conn = engine.connect()
        data = pd.read_sql(sql, conn)
        conn.close()

        data = data.drop(columns=['Id'])

        data.loc[data['Status'].isna(), 'Status'] = data.loc[data['Status'].isna(), 'Status'].fillna('Bulutlu')

        # null kontrolü
        def get_missing(x):
            return (sum(x.isnull()))

        missings = data.apply(get_missing, axis=0)
        fill_missing = missings[missings > 0]
        for i in fill_missing.index:
            data[i] = data[i].fillna(0)

        data = pd.get_dummies(data, columns=['bolge', 'weekday', 'Status'], drop_first=False)
        data = pd.get_dummies(data, columns=['hotel_types'], drop_first=True)

        columns_to_numeric = ['avg3click_hotelimpr', 'avg7click_hotelimpr', 'avg30click_hotelimpr', 'booking_value_index', ]

        for i in columns_to_numeric:
            data[i] = pd.to_numeric(data[i], errors='ignore')

        yesteday = datetime.today() - timedelta(days=1)
        max_date1 = yesteday.strftime("%Y-%m-%d")
        test = data.loc[data['log_date'] == str(max_date1)]
        data = data.loc[data['log_date'] < str(max_date1)]
        max_date = max(data.log_date)
        train = data.loc[data['log_date'] < str(max_date)]

        logs_ctr['log_date'] = str(max_date1)
        logs_impr['log_date'] = str(max_date1)

        # save copies of train and test sets for CTR prediction
        test_impr = test.copy()
        train_impr = train.copy()

        # %% ## CTR Prediction  -------------------------------------------------------------------------------------------------
        delete = ['avg_cpc', 'clicks', 'hotel_impr', 'trivago_id', 'log_date']

        train_ctr = train.drop(delete, axis=1)
        test_ctr = test.drop(delete, axis=1)

        # set the target variable for training set
        X_ctr = train_ctr.drop('click_hotel_impr', axis=1)  # click_hotel_impr = CTR
        y_ctr = train_ctr['click_hotel_impr']

        # set the target variable for test set
        X_test_ctr = test_ctr.drop('click_hotel_impr', axis=1)
        y_test_ctr = test_ctr['click_hotel_impr']

        # validation data for score calculation
        X_train, X_val, y_train, y_val = train_test_split(X_ctr, y_ctr, test_size=0.25, random_state=5)

        print('\n CTR ***************** training for validation scores\n')
        xgb = XGBRegressor(learning_rate=0.1, max_depth=8, n_estimators=200, n_jobs=njobs)
        xgb.fit(X=X_train, y=y_train, eval_metric=['rmse'])
        y_pred = xgb.predict(X_val)

        r2_score = r2(y_val, y_pred)
        mse_score = mse(y_val, y_pred)
        mae_err = mae(y_val, y_pred)

        print('r2:', r2_score)
        print('mse:', mse_score)
        print('mae:', mae_err)

        logs_ctr['r2_validation'] = r2_score
        logs_ctr['mse_validation'] = mse_score
        logs_ctr['mae_validation'] = mae_err

        y_val_pred_ctr = y_pred.copy()

        s_test = y_val.sum()
        s_pred = y_pred.sum()
        if s_pred > s_test:
            print('Sum total sales real:pred -', float(s_test) / float(s_pred))
            logs_ctr['sum_cost_validation'] = float(s_test) / float(s_pred)
            logs_ctr['sum_validation_is_pred_bigger'] = 1
        else:
            print('Sum total sales pred:real -', float(s_pred) / float(s_test))
            logs_ctr['sum_cost_validation'] = float(s_pred) / float(s_test)
            logs_ctr['sum_validation_is_pred_bigger'] = 0

        ## retrain with full data for live prediction
        xgb_ctr_full = XGBRegressor(learning_rate=0.1, max_depth=8, n_estimators=200, n_jobs=njobs)
        xgb_ctr_full = xgb_ctr_full.fit(X_ctr, y_ctr)
        y_predCTR = xgb_ctr_full.predict(X_test_ctr)

        print('Test day:', max_date, 'only ---------------------------')
        print('r2:', r2(y_test_ctr, y_predCTR))
        logs_ctr['r2_test_day'] = r2(y_test_ctr, y_predCTR)
        print('mse:', mse(y_test_ctr, y_predCTR))
        logs_ctr['mse_test_day'] = mse(y_test_ctr, y_predCTR)
        print('mae:', mae(y_test_ctr, y_predCTR))
        logs_ctr['mae_test_day'] = mae(y_test_ctr, y_predCTR)
        s_test = y_test_ctr.sum()
        s_pred = y_predCTR.sum()
        if s_pred > s_test:
            print('Sum total sales real:pred -', float(s_test) / float(s_pred))
            logs_ctr['sum_cost_test_day'] = float(s_pred) / float(s_test)
            logs_ctr['sum_test_day_is_pred_bigger'] = 1
        else:
            print('Sum total sales pred:real -', float(s_pred) / float(s_test))
            logs_ctr['sum_cost_test_day'] = float(s_pred) / float(s_test)
            logs_ctr['sum_test_day_is_pred_bigger'] = 0

        # start preparing data that will be inserted into database
        y_predCTR[y_predCTR < 0] = 0

        output = test[['trivago_id', 'log_date', 'clicks']].copy()
        output['pred_CTR'] = y_predCTR

        # %% Hotel Impression ---------------------------------------------------------------------------------------------------
        delete = ['avg_cpc', 'clicks', 'click_hotel_impr', 'trivago_id', 'log_date']
        train_impr = train_impr.drop(delete, axis=1)
        test_impr = test_impr.drop(delete, axis=1)

        # setting target varible for train and test datasets
        X_impr = train_impr.drop('hotel_impr', axis=1)  # click_hotel_impr = CTR
        y_impr = train_impr['hotel_impr']

        X_test_impr = test_impr.drop('hotel_impr', axis=1)
        y_test_impr = test_impr['hotel_impr']

        # validation data for score calculation
        X_train, X_val, y_train, y_val = train_test_split(X_impr, y_impr, test_size=0.25, random_state=5)

        print('\nHotel Impression ***************** training for validation scores\n')
        xgb2 = XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=500, gamma=20, n_jobs=njobs)
        xgb2.fit(X=X_train, y=y_train, eval_metric=['rmse'])
        y_pred = xgb2.predict(X_val)

        r2_score = r2(y_val, y_pred)
        mse_score = mse(y_val, y_pred)
        mae_err = mae(y_val, y_pred)
        print('r2:', r2_score)
        print('mse:', mse_score)
        print('mae:', mae_err)
        logs_impr['r2_validation'] = r2_score
        logs_impr['mse_validation'] = mse_score
        logs_impr['mae_validation'] = mae_err

        y_val_pred_impr = y_pred.copy()

        s_test = y_val.sum()
        s_pred = y_pred.sum()
        if s_pred > s_test:
            print('Sum total sales real:pred -', float(s_test) / float(s_pred))
            logs_impr['sum_cost_validation'] = float(s_test) / float(s_pred)
            logs_impr['sum_validation_is_pred_bigger'] = 1
        else:
            print('Sum total sales pred:real -', float(s_pred) / float(s_test))
            logs_impr['sum_cost_validation'] = float(s_pred) / float(s_test)
            logs_impr['sum_validation_is_pred_bigger'] = 0

        ## retrain with full data for live prediction
        xgb_impr_full = XGBRegressor(learning_rate=0.1, max_depth=8, n_estimators=200, n_jobs=njobs)
        xgb_impr_full = xgb_impr_full.fit(X_impr, y_impr)
        y_predIMPR = xgb_impr_full.predict(X_test_impr)

        print('Test day:', max_date, 'only ---------------------------')
        print('r2:', r2(y_test_impr, y_predIMPR))
        logs_impr['r2_test_day'] = r2(y_test_impr, y_predIMPR)
        print('mse:', mse(y_test_impr, y_predIMPR))
        logs_impr['mse_test_day'] = mse(y_test_impr, y_predIMPR)
        print('mae:', mae(y_test_impr, y_predIMPR))
        logs_impr['mae_test_day'] = mae(y_test_impr, y_predIMPR)
        s_test = y_test_impr.sum()
        s_pred = y_predIMPR.sum()
        if s_pred > s_test:
            print('Sum total sales real:pred -', float(s_test) / float(s_pred))
            logs_impr['sum_cost_test_day'] = float(s_test) / float(s_pred)
            logs_impr['sum_test_day_is_pred_bigger'] = 1
        else:
            print('Sum total sales pred:real -', float(s_pred) / float(s_test))
            logs_impr['sum_cost_test_day'] = float(s_pred) / float(s_test)
            logs_impr['sum_test_day_is_pred_bigger'] = 0

        y_predIMPR[y_predIMPR < 0] = 0

        output['pred_impr'] = y_predIMPR
        output['pred_click'] = np.round(y_predIMPR * y_predCTR, 0).astype(int)

        logs_ctr['has_weather_status'] = 2
        logs_impr['has_weather_status'] = 2

        y_clicks = train.loc[y_val.index, 'clicks']
        y_pred_val_clicks = np.round(y_val_pred_ctr * y_val_pred_impr, 0).astype(int)
        y_pred_val_clicks[y_pred_val_clicks < 0] = 0

        logs_clicks = {'log_date': max_date1, 'prediction_type': 'clicks', 'has_weather_status': 2}
        logs_clicks['r2_validation'] = r2(y_clicks, y_pred_val_clicks)
        logs_clicks['mse_validation'] = mse(y_clicks, y_pred_val_clicks)
        logs_clicks['mae_validation'] = mae(y_clicks, y_pred_val_clicks)

        s_test = y_clicks.sum()
        s_pred = y_pred_val_clicks.sum()
        if s_pred > s_test:
            logs_clicks['sum_cost_validation'] = float(s_test) / float(s_pred)
            logs_clicks['sum_validation_is_pred_bigger'] = 1
        else:
            logs_clicks['sum_cost_validation'] = float(s_pred) / float(s_test)
            logs_clicks['sum_validation_is_pred_bigger'] = 0

        y_test_clicks = test['clicks']

        logs_clicks['r2_test_day'] = r2(y_test_clicks, output['pred_click'])
        logs_clicks['mse_test_day'] = mse(y_test_clicks, output['pred_click'])
        logs_clicks['mae_test_day'] = mae(y_test_clicks, output['pred_click'])

        s_test = y_test_clicks.sum()
        s_pred = output['pred_click'].sum()
        if s_pred > s_test:
            logs_clicks['sum_cost_test_day'] = float(s_test) / float(s_pred)
            logs_clicks['sum_test_day_is_pred_bigger'] = 1
        else:
            logs_clicks['sum_cost_test_day'] = float(s_pred) / float(s_test)
            logs_clicks['sum_test_day_is_pred_bigger'] = 0

        logs_ctr_mail = logs_ctr.copy()
        importants_ctr = pd.DataFrame(
            {'feature': X_ctr.columns, 'importance': np.round(xgb_ctr_full.feature_importances_, 5)})
        importants_ctr = importants_ctr.sort_values('importance', ascending=False).reset_index()
        logs_ctr['feature_importances'] = importants_ctr.iloc[:15, 1].to_json()
        logs_ctr_mail['feature_importances'] = importants_ctr.iloc[:15, 1].to_dict()

        logs_impr_mail = logs_impr.copy()
        importants_impr = pd.DataFrame(
            {'feature': X_impr.columns, 'importance': np.round(xgb_impr_full.feature_importances_, 5)})
        importants_impr = importants_impr.sort_values('importance', ascending=False).reset_index()
        logs_impr['feature_importances'] = importants_impr.iloc[:15, 1].to_json()
        logs_impr_mail['feature_importances'] = importants_impr.iloc[:15, 1].to_dict()

        # # save predictions and logs to database
        conn = engine.connect()
        output.to_sql('ctr_impr_prediction_all_positions', conn, if_exists='append', index=False)
        (pd.DataFrame([logs_ctr, logs_impr, logs_clicks])).to_sql('model_logs', conn, if_exists='append', index=False)
        conn.close()
        print('Wrote results to tables for', max_date1)

        fromaddr = "bilgilendirme@cerebro.tech"
        toaddr = "datateam@cerebro.tech"
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = "Click Predictions with All Positions"

        body = "Click prediction worked!\n" + json.dumps(logs_clicks, indent=4) + "\nCTR Results\n" + \
               json.dumps(logs_ctr_mail, indent=4) + "\nImpression Results\n" + json.dumps(logs_impr_mail, indent=4)
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(fromaddr, "CeReBrO!*")
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        server.quit()

    except Exception as e:
        print(e)
        fromaddr = "bilgilendirme@cerebro.tech"
        toaddr = "datateam@cerebro.tech"
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = "Click Predictions with All Positions FAIL"

        body = "Click prediction failed!\n" + str(e)
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(fromaddr, "CeReBrO!*")
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        server.quit()


try:
    x=datetime.today()
    # y=x.replace(day=x.day+1, hour=14, minute=0, second=0, microsecond=0
    y = x + timedelta(days=1)
    y = y.replace(hour=14, minute=0)
    delta_t=y-x
    secs=delta_t.seconds+1
    t = Timer(secs, click_prediction_job)
    t.start()
except Exception as e:
    print(e)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("bilgilendirme@cerebro.tech", "CeReBrO!*")
    msg = "Something failed!\n" + str(e)
    server.sendmail("bilgilendirme@cerebro.tech", "gizem.aras@cerebro.tech", msg)
    server.quit()


# click_prediction_job()