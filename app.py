import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask
from flask import render_template
from sklearn import preprocessing
from datetime import datetime
from datetime import timedelta

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def root():

        results = []
        # ---------- read CSV with required no of cols and rows -----------
        reader= pd.read_csv("DMS_A_OUTAGE_REPAIR.csv", usecols=['OUTAGE_NO','TIME_STAMP','DISTRICT_NO','MAX_CUST_AFF','DEVCAT','EST_RES_DATE'], nrows=50)
        
        #Converting timestamp to datetime type: Not required now
        #reader['TIME_STAMP'] = pd.to_datetime(reader['TIME_STAMP'])
        
        records = reader.to_records(index=False)
        headings = list(reader.columns)
        #headings.append('EST_REPAIR_TIME')
        data = list(records)
        # ---------- convert device category from numeric to category -----------
        DEVCAT_dict = {
                0 :  'Absent',
                1 :  'Site',
                2 :  'Node',
                3 :  'Line',
                4 :  'Switch (state unknown or immaterial)',
                6 :  'Transformer or Regulator',
                7 :  'Capacitor',
                8 :  'Load',
                9 :  'Source',
                16 :  'Line Device i.e. switch, protector or regulator',
                17 :  'Circuit Breaker (previously relay)',
                18 :  'Fuse',
                19 :  'Electronic Recloser',
                26 :  'Line Cut',
                36 :  'Metered point',
                37 :  'Path on geographic',
                38 :  'Path on schematic',
                40 :  'Leader Lines',
                71 :  'User 1 Device',
                72 :  'User 2 Device',
                82 :  'Line Drop Compensator',
                83 :  'Motor, electric',
                90 :  'Fault Indicator',
                92 :  'Customer',
                93 :  'Maximum Line Device Category Number'
        }

        return render_template('home.html', headings=headings, data=data, DEVCAT_dict=DEVCAT_dict)




@app.route('/my-link/', methods=['POST', 'GET'])
def my_link():
  reader_new= pd.read_csv("DMS_A_OUTAGE_REPAIR.csv")
  predictionModel(reader_new)
  return root() 
                            

 
def predictionModel(df):

    vectorizer = joblib.load('vectorizer.pkl')
    rf_model = joblib.load('rf_model.pkl')
    gb_model = joblib.load('gb_model.pkl')
    
    #WHAT IF TIME STAMP IS ALSO NULL. WHAT ERROR MESSAGES WILL BE GIVEN THEN
    
    #Features cleaning and filling up null valuesB  B V VVE3
    #For date type features, if their values are not present, make them same as the time stamp
    df["AFRM_DATE"].fillna(df["TIME_STAMP"], inplace=True)
    df["DIAG_DATE"].fillna(df["TIME_STAMP"], inplace=True)
    df["CUR_DEV_TIMESTAMP"].fillna(df["TIME_STAMP"], inplace=True)

    
    #Adding Feature engineering
    df['FE_OUTAGE_DATE'] = pd.to_datetime(df['TIME_STAMP']).dt.date  
    df['FE_OUTAGE_TIME'] = pd.to_datetime(df['TIME_STAMP']).dt.time  
    df['FE_OUTAGE_DAY'] = pd.to_datetime(df['TIME_STAMP']).dt.day  
    df['FE_OUTAGE_MONTH'] = pd.to_datetime(df['TIME_STAMP']).dt.month
    df['FE_OUTAGE_DOW'] = pd.to_datetime(df['TIME_STAMP']).dt.dayofweek
    df['FE_DIAGNOSE_DATE'] = pd.to_datetime(df['DIAG_DATE']).dt.date  
    df['FE_DIAGNOSE_TIME'] = pd.to_datetime(df['DIAG_DATE']).dt.time  
    df['FE_DIAGNOSE_DAY'] = pd.to_datetime(df['DIAG_DATE']).dt.day  
    df['FE_DIAGNOSE_MONTH'] = pd.to_datetime(df['DIAG_DATE']).dt.month
    df['FE_AFRM_DATE_DATE'] = pd.to_datetime(df['AFRM_DATE']).dt.date  
    df['FE_AFRM_DATE_TIME'] = pd.to_datetime(df['AFRM_DATE']).dt.time  
    df['FE_AFRM_DATE_DAY'] = pd.to_datetime(df['AFRM_DATE']).dt.day  
    df['FE_AFRM_DATE_MONTH'] = pd.to_datetime(df['AFRM_DATE']).dt.month
    df['FE_CUR_DEV_DATE'] = pd.to_datetime(df['CUR_DEV_TIMESTAMP']).dt.date  
    df['FE_CUR_DEV_TIME'] = pd.to_datetime(df['CUR_DEV_TIMESTAMP']).dt.time  
    df['FE_CUR_DEV_DAY'] = pd.to_datetime(df['CUR_DEV_TIMESTAMP']).dt.day  
    
    #Remove all extra features: 
    del_features = ['OUTAGE_NO', 'EST_RES_DATE', 'TIME_STAMP', 'TIME_MOD', 'DIAG_DATE', 'REST_DATE', 'REST_CALLBACK_DONE', 'REST_CALLBACK_DATE', 'EST_REPAIR_TIME', 'EST_REPAIR_EXPIRY', 'CURRENT_EST_REPAIR_TIME', 'REGION_NO', 'DISTRICT_NO', 'DEV_ID', 'ADDRESS', 'DEV_NAME', 'SG_KEY', 'PRIMARY_RPT', 'NO_CALLS', 'CUST_AFF', 'CRIT_CUST', 'KVA_AFF', 'CUST_HRS', 'RPTD_DONE', 'RPTD_DATE', 'RPTD_COMT', 'DIAG_DONE', 'DIAG_COMT', 'DISP_DONE', 'DISP_DATE', 'DISP_COMT', 'ARIV_DONE', 'ARIV_DATE', 'ARIV_COMT', 'CAUS_DONE', 'CAUS_DATE', 'CAUS_COMT', 'RPAR_DONE', 'RPAR_DATE', 'RPAR_COMT', 'REST_DONE', 'REST_COMT', 'VERF_DONE', 'VERF_DATE', 'VERF_COMT', 'PHASES', 'AFRM_DONE', 'AFRM_DATE', 'AFRM_COMT', 'HAZARD', 'WORK_ORDER_NO', 'OUTAGE_TYPE', 'EST_OVERRIDE', 'DEV_FPOS', 'CUR_CUST_AFF', 'TOTAL_CUST_AFF', 'CUST_AFF_CHG_DATE', 'PRINT_DATE', 'AVG_MIN_OUT', 'ATTN', 'NEEDS_UPDATE_PRIME_RPT', 'COMMENT_CAT_NO', 'OUTAGE_OAR_CODE', 'LINE_ID', 'LINE_NAME', 'FROZEN_FLAG', 'PEND_DONE', 'PEND_DATE', 'METER_ROUTE', 'METER_CYCLE', 'OUTAGE_COMMENT', 'SUB_AREA', 'NEEDS_CUSTMIN_RECALC', 'SY_STATUS', 'WAS_ORPHANED', 'INCD_DONE', 'INCD_DATE', 'ARCH_DONE', 'ARCH_DATE', 'ABB_INT_ID', 'OP_CENTER_ID', 'IN_SYSTEM', 'JOB_NO', 'OUTAGE_TYPE_KEY', 'BPHASES', 'RELATED_EVENT', 'ITR_EXPIRED_TIME', 'ETR_EXPIRED_TIME', 'MASTER_OUTAGE_NO', 'CBOK_METHOD', 'SECONDARY', 'ORDER_NUMBER', 'SPID', 'WAS_PLANNED', 'REQC_DONE', 'REQC_DATE', 'ALT_DEVICE_FPOS', 'ALT_DEVCAT', 'ALT_DEV_ID', 'RESTORATION_VERIFY', 'KEY_CALL_COMMENTS', 'OP_COMP', 'PRIORITY', 'KEY_CALL_NO', 'PERSONNEL_STANDING_BY', 'EXTERNAL_ID', 'ASSET_DATA_ERROR_FLAG', 'ACK_DONE', 'ACK_DATE', 'CUR_DEV_TIMESTAMP', 'FE_KEY', 'TEST_MODE', 'SERVICE_CODE', 'BUILD_TYPE', 'BLOCK_MOBILE_OUTBOUND', 'WORK_TAG_COUNT', 'SCADA_INITIATED', 'DERIVED_PRIORITY', 'PRIORITY_OVERRIDE','ACCNO', 'OPERATOR']
    for i, item in enumerate(del_features):
        del df[item];
        
        
    #Replacement of null values of features where null values were present
    df['HAZARD_LEVEL_CODE'].fillna(-1, inplace = True)
    df['STAT_CALL'].fillna(0, inplace = True)
    df['HAZARD_COUNT'].fillna(-1, inplace = True)
    
    #Label encoding of each feature
    label_encoder =  preprocessing.LabelEncoder()

    df['CIVIL_DISTR_NAME'] = label_encoder.fit_transform(df['CIVIL_DISTR_NAME'])
    df['CIVIL_DISTR_NAME'].astype('int64')

    #DEV_CAT
    df['DEV_CAT'] = label_encoder.fit_transform(df['DEV_CAT'])
    df['DEV_CAT'].astype('int64')


    #MOMENTARY_OUTAGE : Categorical      
    df['MOMENTARY_OUTAGE'] = label_encoder.fit_transform(df['MOMENTARY_OUTAGE'])
    df['MOMENTARY_OUTAGE'].astype('int64')



    #FE_OUTAGE_DATE  
    df['FE_OUTAGE_DATE'] = label_encoder.fit_transform(df['FE_OUTAGE_DATE'])
    df['FE_OUTAGE_DATE'].astype('int64')

    #FE_OUTAGE_TIME         
    df['FE_OUTAGE_TIME'] = label_encoder.fit_transform(df['FE_OUTAGE_TIME'])
    df['FE_OUTAGE_TIME'].astype('int64')

    df['FE_DIAGNOSE_DATE'] = label_encoder.fit_transform(df['FE_DIAGNOSE_DATE'])
    df['FE_DIAGNOSE_DATE'].astype('int64')

    df['FE_DIAGNOSE_TIME'] = label_encoder.fit_transform(df['FE_DIAGNOSE_TIME'])
    df['FE_DIAGNOSE_TIME'].astype('int64')

    df['FE_AFRM_DATE_DATE'] = label_encoder.fit_transform(df['FE_AFRM_DATE_DATE'])
    df['FE_AFRM_DATE_DATE'].astype('int64')

    df['FE_AFRM_DATE_TIME'] = label_encoder.fit_transform(df['FE_AFRM_DATE_TIME'])
    df['FE_AFRM_DATE_TIME'].astype('int64')

    df['FE_CUR_DEV_DATE'] = label_encoder.fit_transform(df['FE_CUR_DEV_DATE'])
    df['FE_CUR_DEV_DATE'].astype('int64')

    df['FE_CUR_DEV_TIME'] = label_encoder.fit_transform(df['FE_CUR_DEV_TIME'])
    df['FE_CUR_DEV_TIME'].astype('int64')
    
    #Vectorizing the Ouatage comment feature
    outage_cmnt_features = vectorizer.transform(df['OUTAGE_CMNT'])
    comment_df = pd.DataFrame(outage_cmnt_features.toarray(), columns=vectorizer.get_feature_names())
    selected_Features = comment_df.filter(['118', '18', '54', 'and', 'comp', 'crew', 'down', 'information', 'line', 'need', 'ok', 'on', 'open', 'operation', 'p5429', 'pri', 'primary', 'test', 'to', 'tree', 'unknown'], axis=1)
    
    #Since outage comments are now vectorized, deleting the comment feature
    del df['OUTAGE_CMNT']

     #Merge only certain columns of comment dataframe to the original dataframe
    df_total = pd.concat([df, selected_Features], axis=1)
    
     #Deleting all extra columns introduced
    for count,ele in enumerate(df_total.columns):
        if 'Unnamed' in ele:
            del df_total[ele]
    
    #Predicting the data as per each model now
    gb_predictions = gb_model.predict(df_total) 
    rf_predictions = rf_model.predict(df_total)
    
    #Predictions from both the models is available now
    final_predictions = []
    for i in range(len(rf_predictions)):
        if(gb_predictions[i] > rf_predictions[i]):
            final_predictions.append(gb_predictions[i])
        else:
            final_predictions.append(rf_predictions[i])
    
    
    reader = pd.read_csv("DMS_A_OUTAGE_REPAIR.csv")
    #Removing the pre existing csv now
    os.remove("DMS_A_OUTAGE_REPAIR.csv")
    #Converting hours to Time format as given in time stamp
    
    final_time_array = []
    for count,ele in enumerate(reader['TIME_STAMP']):
        time_str = ele
        date_format_str = '%m/%d/%Y %H:%M'
        given_time = datetime.strptime(time_str, date_format_str)
        n = final_predictions[count]
        final_time = given_time + timedelta(minutes=n)
        final_time_str = final_time.strftime('%m/%d/%Y %H:%M')
        final_time_array.append(final_time_str)

    reader['EST_RES_DATE'] = final_time_array
    #Finally converting it into CSV format  
    reader.to_csv('DMS_A_OUTAGE_REPAIR.csv')

if __name__ == '__main__':
    app.run(debug=False)
