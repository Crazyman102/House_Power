import os
import os.path as osp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
# import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

from model import create_model
from utils import plot_loss, plot_result

def sliding_window (df, n_past, n_future):
    X,y = list(),list()
    for start in range(len(df)):
        end_index = start + n_past
        if end_index + n_future >len(df):
            break
        X.append(df.iloc[start:end_index,:])
        y.append(df.iloc[end_index:end_index+n_future,0])
    return np.array(X), np.array(y)


if __name__=="__main__":
    # read csvdata
    filepath = 'dataset/household_power_consumption.txt'
    df = pd.read_csv(filepath, sep=';',
                    low_memory=False,
                    na_values=['nan','?'],
                    infer_datetime_format=True,
                    parse_dates={'datetime':[0,1]},
                    index_col='datetime')

    # fill mean to null data
    df = df.fillna(df.mean())

    # data resample 15 minutes
    df_reframe = df.resample('15T').mean()

    # data MinMax
    scaler = MinMaxScaler(feature_range=(0,1)).fit(df_reframe.values)
    scaled = scaler.transform(df_reframe)

    # split data 
    past, outsteps, feature = 192,5, 7
    df_reframe = pd.DataFrame(scaled)
    train_split = int(df_reframe.shape[0] * 0.8)
    train_X, train_y = sliding_window(df_reframe.iloc[:train_split, :], past, outsteps)
    test_X, test_y = sliding_window(df_reframe.iloc[train_split:, :], past, outsteps)

    # create model
    name = "LSTM" # options = ["CNN", "LSTM", "CNN_LSTM", "Multi_CNN","multi_CNN_LSTM_Model",'Multi_CNN_cr_LSTM']
    model = create_model(name, past, feature, outsteps)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    plot_model(model, to_file='convolutional_neural_network.png')
    # set path
    savefile_path = f"result/{name}_{past}"
    if not osp.exists(savefile_path):
        os.makedirs(savefile_path)

    # training model
    path = f'result/{name}_{past}/{name}_n{past}_n{outsteps}.h5'
    cp_callback = ModelCheckpoint(path, save_weights_only=True, verbose=2, monitor='mse', mode='min')
    callback_list=[cp_callback]
    # history = model.fit(train_X, train_y, epochs=20, batch_size=32, validation_split=0.2, verbose=1, callbacks=callback_list)
    history = model.fit(train_X, train_y, epochs=20, batch_size=32, validation_split=0.2, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
    

    #  save model
    model.save(path)
    savepic_path = savefile_path+'/{name}_loss.png'
    plot_loss(history.history['loss'], history.history['val_loss'],savefile_path)
