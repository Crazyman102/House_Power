from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from keras import models


def get_testdata(df, in_steps, out_steps, month):
    scaler = MinMaxScaler(feature_range=(0,1)).fit(df.values)       
    test_X = scaler.transform(df[f'2010-{month}-07':f'2010-{month}-09']).reshape(1, in_steps, 7)
    actual_y = df.loc[f'2010-{month}-10'].iloc[:,0].head(int(out_steps))
    return test_X, actual_y, scaler

def get_model(name, n_out, n_in):    
    path =f'result/future{n_out}/{name}_n{n_in}_n{n_out}.h5'
    # path ='result/multi_CNN_LSTM_192/multi_CNN_LSTM_n192_n10.h5'
    model = models.load_model(path)
    return model

def inverse_predict(pred_y, scaler):
    result = np.concatenate((pred_y.flatten().reshape(-1,1),np.zeros((pred_y.flatten().shape[0],6)) ),axis=1)
    result_scaled = scaler.inverse_transform(result)[:,0].reshape(-1, pred_y.shape[1])
    return result_scaled.flatten()

def plot_result(predict_y, actual_y, timesteps, pic_title, pic_path, mae):
    times = list(range(1,timesteps+1))
    plt.figure(figsize=(8,4))
    plt.plot(times,predict_y,'red', marker='.',label='predict')
    plt.plot(times, actual_y, 'blue', marker='.', label='actual')
    plt.title(f'{pic_title}, MAPE:{mae:.3f}')
    plt.legend()
    # plt.show()
    
    plt.savefig(pic_path,bbox_inches='tight')
    return

df = read_csv('dataset/household_power_consumption.txt',sep=';',
            low_memory=False,na_values=['nan','?'],
            infer_datetime_format=True,             
            parse_dates={'datetime':[0,1]},
            index_col='datetime')

df_clear = df.fillna(df.mean())
df_clear = df.resample('15T').mean()

# Multi_CNN, CNN, LSTM,CNN_LSTM,multi_CNN_LSTM
model_name, out_steps, in_steps = 'Multi_CNN', '5', '288'

model = get_model(model_name, out_steps, in_steps)

mae,mape=[],[]
for i in range(1,12):   

    # prdict value
    test_X, y_actual, trans_scaler = get_testdata(df_clear, int(in_steps), out_steps, i)
    pred_y = model.predict(test_X)
    inve_y = inverse_predict(pred_y, trans_scaler)
    
    # title label
    title = f'Model:{model_name} in_steps:{in_steps} outsteps:{out_steps}  Month:{i}'
    picpath = f'result/pic/{model_name}_{i}.png'
    cal_mae = mean_absolute_error(y_actual, inve_y)
    cal_mape = mean_absolute_percentage_error(y_actual, inve_y)*100
    plot_result(inve_y, y_actual, int(out_steps), title,picpath, cal_mape)
    mae.append(f'{cal_mae:.3f}')
    mape.append(f'{cal_mape:.3f}')


