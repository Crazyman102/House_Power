from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D,Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate


def create_model(name, timesteps, feature, outsteps):
    if name == "CNN":
        return CNN_Model(timesteps, feature, outsteps)
    elif name == "LSTM":
        return LSTM_Model(timesteps, feature, outsteps)
    elif name == "CNN_LSTM":
        return CNN_LSTM_Model(timesteps, feature, outsteps)
    elif name == "Multi_CNN":
        return Multi_CNN_Model(timesteps, feature, outsteps)
    elif name == "multi_CNN_LSTM":
        return multi_CNN_LSTM_Model(timesteps, feature, outsteps)
    elif name == "Multi_CNN_cr_LSTM":
        return Multi_CNN_cr_LSTM_Model(timesteps, feature, outsteps)
    else:
        raise f"The choosing model {name}, not exit."

def CNN_Model(timesteps, feature, n_out):
    model = Sequential()
    model.add(Conv1D(filters=32,kernel_size=4,activation='relu',input_shape=(timesteps,feature)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16,kernel_size = 2, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(96,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_out))
    return model

def LSTM_Model(timesteps, feature, n_out):
    model = Sequential()
    # model.add(LSTM(100, input_shape=(timesteps,feature)))
    # model.add(Dropout(0.2))
    # model.add(Dense(n_out))  
    model.add(LSTM(96, return_sequences = True,input_shape=(timesteps,feature)))
    model.add(Dropout(0.1))
    model.add(LSTM(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(n_out))   
    return model

def CNN_LSTM_Model(timesteps, feature, n_out):
    model = Sequential()        
    model.add(Conv1D(filters=32,
               kernel_size=4,
               activation='relu',
               padding='same', input_shape = (timesteps,feature)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))       
    model.add(LSTM(10, activation='relu'))
    model.add(Dense(n_out))    
    return model

def Multi_CNN_Model(timesteps, feature, n_out):
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
    output = Dense(n_out)(hidden3_out)
    model = Model(inputs = visible, outputs=output)   
    return model

def multi_CNN_LSTM_Model(timesteps, feature, n_out):
    inputs = Input(shape=(timesteps, feature))
    # first model
    conv1 = Conv1D(filters = 32,kernel_size=8,activation='relu')(inputs)
    conv1_drop = Dropout(0.1)(conv1)
    conv1_2 = Conv1D(filters= 16,kernel_size=8,activation='relu')(conv1_drop)
    flat1 = Flatten()(conv1_2)

    # second model 
    lstm1 = LSTM(64,activation='relu', return_sequences=True)(inputs)
    drop1= Dropout(0.1)(lstm1)
    lstm1_2 =LSTM(32,activation='relu')(drop1)
    dense0 =Dense(96,activation='relu')(lstm1_2)
    merge =concatenate([flat1,dense0])
    dense1 = Dense(192, activation = 'relu')(merge)
    dense2 = Dense(96,activation='relu')(dense1)
    dense3 = Dense(48,activation='relu')(dense2)
    dense4 = Dense(24,activation='relu')(dense3)
    outputs = Dense(n_out)(dense4)

    model = Model(inputs = inputs, outputs = outputs)   
    return model


def Multi_CNN_cr_LSTM_Model(timesteps, feature, n_out):
    inputs =Input(shape=(timesteps, feature))

    conv1 = Conv1D(filters= 32,kernel_size=8, activation='relu')(inputs)
    conv1_drop = Dropout(0.2)(conv1)
    conv1_2 = Conv1D(filters= 16,kernel_size=8,activation='relu')(conv1_drop)
    lstm1 = LSTM(64,activation='relu', return_sequences=True)(conv1_2)
    lstm_drop1 = Dropout(0.1)(lstm1)
    lstm1_2 = LSTM(32, activation='relu')(lstm_drop1)
    dense1= Dense(64, activation='relu')(lstm1_2)
    

    conv2 = Conv1D(filters= 16,kernel_size=8, activation='relu')(inputs)
    conv2_drop = Dropout(0.2)(conv2)
    conv2_2 = Conv1D(filters= 8,kernel_size=4,activation='relu')(conv2_drop)
    lstm2 = LSTM(64,activation='relu', return_sequences=True)(conv2_2)
    lstm_drop2 = Dropout(0.1)(lstm2)
    lstm2_2 = LSTM(32, activation='relu')(lstm_drop2)
    dense2= Dense(64, activation='relu')(lstm2_2)

    conv3 = Conv1D(filters= 8,kernel_size=2, activation='relu')(inputs)
    conv3_drop = Dropout(0.2)(conv3)
    conv3_2 = Conv1D(filters= 4,kernel_size=2,activation='relu')(conv3_drop)
    lstm3 = LSTM(64,activation='relu', return_sequences=True)(conv3_2)
    lstm_drop3 = Dropout(0.1)(lstm3)
    lstm3_2 = LSTM(32, activation='relu')(lstm_drop3)
    dense3= Dense(64, activation='relu')(lstm3_2)

    merge =concatenate([dense1, dense2, dense3])

    dense_a1 = Dense(96,activation='relu')(merge)
    dense_a2 = Dense(48,activation='relu')(dense_a1)
    dense_a2 = Dense(24,activation='relu')(dense_a2)
    outputs = Dense(n_out)(dense_a2)

    model = Model(inputs = inputs, outputs = outputs)   
    return model
