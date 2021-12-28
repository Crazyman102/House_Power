from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns

# for deep learning
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D,Input
from keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate
import os

from sys import exit

# function 
def sliding_window (df, n_past, n_future):
    X,y = list(),list()
    for start in range(len(df)):
        end_index = start + n_past
        if end_index + n_future >len(df):
            break
        X.append(df.iloc[start:end_index,:])
        y.append(df.iloc[end_index:end_index+n_future,0])
    return np.array(X), np.array(y)

def CNN_Model(timesteps, feature):
    model = Sequential()
    model.add(Conv1D(filters=32,kernel_size=4,activation='relu',input_shape=(timesteps,feature)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16,kernel_size = 2, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(96,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def LSTM_Model(timesteps, feature):
    model = Sequential()
    model.add(LSTM(96, return_sequences = True,input_shape=(timesteps,feature)))
    model.add(Dropout(0.2))
    model.add(LSTM(48, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(10))
    model.compile( optimizer='adam', loss='mse',metrics=['mae'])
    return model

def CNN_LSTM_Model(timesteps, feature):
    model = Sequential()        
    model.add(Conv1D(filters=32,
               kernel_size=4,
               activation='relu',
               padding='same', input_shape = (timesteps,feature)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))       
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(5))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])    
    return model

def Multi_CNN_Model(timesteps, feature):
    visible = Input(shape=(timesteps, feature))
    # first model
    conv1 = Conv1D(filters = 32,kernel_size=8,activation='relu')(visible)
    conv1_drop = Dropout(0.2)(conv1)
    conv1_2 = Conv1D(filters= 16,kernel_size=8,activation='relu')(conv1_drop)
    flat1 = Flatten()(conv1_2)
    # second model
    conv2 = Conv1D(filters = 16,kernel_size=4,activation='relu')(visible)
    conv2_drop =Dropout(0.2)(conv2)
    conv2_2 = Conv1D(filters= 8,kernel_size=4,activation='relu')(conv2_drop)
    flat2 = Flatten()(conv2_2)
    # third model
    conv3 = Conv1D(filters = 8, kernel_size=2,activation='relu')(visible)
    conv3_drop =Dropout(0.2)(conv3)
    conv3_2 = Conv1D(filters= 4,kernel_size=2,activation='relu')(conv3_drop)
    flat3 = Flatten()(conv3_2)
    # merge model
    merge =concatenate([flat1,flat2,flat3])
    # hidden layer
    # hidden1 = Dense(672, activation = 'relu')(merge)
    # hidden2 = Dense(384, activation = 'relu')(hidden1)
    hidden2 = Dense(384, activation = 'relu')(merge)
    hidden2_out = Dropout(0.2)(hidden2)
    hidden3 = Dense(192, activation = 'relu')(hidden2_out)
    hidden3_out = Dropout(0.2)(hidden3)
    output = Dense(96)(hidden3_out)
    model =Model(inputs = visible, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
    return model




def plot_result(predict,actual):    
    sample=[1,100,200,400,800]
    for i in sample:
        epoch = range(1,97)
        plt.figure(figsize=(15,10))
        plt.plot(epoch,predict[i,:],'red',marker='.' ,label='predict_value')
        plt.plot(epoch,actual[i,:],'blue', label='actual_value')
        plt.xlabel('epoch',size=15)
        plt.ylabel('value',size=15)
        plt.legend()
        plt.show()
        plt.clf()
        
def plot_loss(loss, val_loss ,savepath):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(savepath+'.png')
    plt.show()

# read csvdata
filepath = 'dataset/household_power_consumption.txt'
df = read_csv(filepath, sep=';',
                low_memory=False,
                na_values=['nan','?'],
                infer_datetime_format=True,
                parse_dates={'datetime':[0,1]},
                index_col='datetime')

# fill mean to null data
df =df.fillna(df.mean())

# data resample 15 minutes
df_reframe = df.resample('15T').mean()

# data MinMax
scaler = MinMaxScaler(feature_range=(0,1)).fit(df_reframe.values)
scaled = scaler.transform(df_reframe)

# split data 
past,future,feature = 672, 10, 7
df_reframe = DataFrame(scaled)
train_split = int(df_reframe.shape[0] * 0.8)
train_X, train_y = sliding_window(df_reframe.iloc[:train_split, :], past, future)
test_X, test_y = sliding_window(df_reframe.iloc[train_split:, :], past, future)

# use model 
# model = CNN_Model(past, feature)
model = LSTM_Model(past, feature)
# model = CNN_LSTM_Model(past, feature)
# model = Multi_CNN_Model(past, feature)
model.summary()

# cnn model
# savefile_path = 'result/CNN_%s'%past
# lstm model
savefile_path = 'result/lstm_%s'%past
# lstm_cnn model
# savefile_path = 'result/lstmCNN_%s'%past
# multi_cnn model
# savefile_path = 'result/multi_CNN_%s'%past
if not os.path.exists(savefile_path):
    os.makedirs(savefile_path)

# training model
# path = 'result/CNN_{0}/cnn_n{1}.h5'.format(past,past)
# path = 'result/CNN_{0}/lstmcnn_n{1}.h5'.format(past,past)
# path = 'result/multi_CNN_{0}/multicnn_n{1}.h5'.format(past,past)
path = 'result/lstm_{0}/lstm_n{1}.h5'.format(past,past)
cp_callback = ModelCheckpoint(path, save_weights_only=True, verbose=2, monitor='mse', mode='min')
callback_list=[cp_callback]
history = model.fit(train_X, train_y, epochs=20, batch_size=32, validation_split=0.2, verbose=1, callbacks=callback_list)
#  save model
model.save(path)
savepic_path = savefile_path+'/lstm_loss.png'
plot_loss(history.history['loss'], history.history['val_loss'],savefile_path)


# # predict data
# from keras import models
# model_path = 'result/CNN_{0}/cnn_n{1}.h5'.format(past,past)
# model = models.load_model(model_path)
# pred_y = model.predict(test_X)

# # inverse data
# #  將預測的shape 維度(sample,96) ==> 維度(sample*96,7)
# result = np.concatenate((pred_y.flatten().reshape(-1,1),np.zeros((pred_y.flatten().shape[0],6)) ),axis=1)
# result_scaled = scaler.inverse_transform(result)[:,0].reshape(-1, pred_y.shape[1])


# y_real = np.concatenate((test_y.flatten().reshape(-1,1),np.zeros((test_y.flatten().shape[0],6)) ),axis=1)
# y_scaled = scaler.inverse_transform(y_real)[:,0].reshape(-1, test_y.shape[1])

# plot_result(result_scaled, y_scaled)