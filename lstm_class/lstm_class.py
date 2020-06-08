import os
import time
import warnings#tensorflowのワーニングの軽減
import numpy as np
import pandas as pd
from numpy import newaxis#配列変更
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential#時系列データ
import matplotlib.pyplot as plt
import normalize_windows


#tensorflow,numpyのエラーを押さえる
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def load_data(filename, seq_len, normalize_window):
#     f = open(filename, 'rb').read()
#     data = f.decode().split('\n')
    
    df = pd.read_csv(filename,index_col=0)#(763, 12)
    df.index=pd.to_datetime(df.index)
    df =np.array(df)
    
    # # reshape input to be [samples, time steps, features]
    sequence_length = seq_len + 1
    result = []
    for k in range(len(df) - sequence_length):#index:0～712
        result.append(df[k: k + sequence_length])
    result = np.array(result)
    print("all_data_shape:{}".format(np.shape(result)))#(712,51,12)
    
    if normalize_window:#window内で標準化
        result = normalize_windows.normalize_windows(result)
    
    result = np.array(result)
    train = result[:500,:,:]#(500,51,12)
    x_train = train[:,:-1,3:]#(500,50,9)
    y_train = train[:,-1,:3]#(500,3)
    x_test = result[500:,:-1,3:]#(212,50,9)
    y_test = result[500:,-1,:3]#(212,3)
    
    print("learning_label_shape：{}".format(np.shape(y_train)))
    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    """
    inputはlayers[0]ここでは50×1が入る
    1番目のLSTM層がlayers[1]
    2番目のLSTM層がlayers[2]
    出力層がlayers[3]でregressionが
    """
    model = Sequential()
    model.add(LSTM(input_shape=(layers[1], layers[0]),
                  output_dim=layers[1],
                  return_sequences=True))
    #Dropoutさせる比率
    model.add(Dropout(0.5))
    
    #2番目のLSTM層
    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.5))
    
    #全結合層
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("softmax"))#活性化関数　通常はsigmoid
    
    #modelに結果が格納
    start = time.time()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    #model.compile(loss="mse",optimizer="rmsprop", metrics =['accuracy'])
    print("実行時間: ",time.time() - start)
    return model 


#model1:一つのデータから一つの株価を予測する
def predict_point_by_point(model, data):
    """
    predicted.sizeは要素数 lenは配列の数
    np.reshapeでnumpy配列に変換
    """
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

#model2:シーケンスから一つを予想　＊理解薄い
def predict_sequence_full(model, data, window_size):
    curr_frame = data[0]#(50,1)
    predicted = []
    for i in range(len(data)):#len(x_test):412
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        #curr_frame[newaxis,:,:] (50,1)→(1,50,1):LSTMに読ませるため　[0,0]でarray形式(1,1)の予測から数値に変換
        curr_frame = curr_frame[1:]#ひとつずらして上書き
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

        #curr_frameの位置[window_size-1]に、predictedの最後の値を,
        #列に沿った処理を行う
        
#model3:50個のシーケンスから、予測シーケンス(prediction_len)の期間予測する　＊理解薄い
def predict_sequences_multiple(model,data,window_size,prediction_len):
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):#予測したいレンジがテストデータに何個含まれているか。その個数でループ。
        curr_frame = data[i*prediction_len]
        predicted =[]
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1],predicted[-1],axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs