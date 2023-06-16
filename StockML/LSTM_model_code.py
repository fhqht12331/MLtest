import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 정규화 함수 
def MinMaxScaler(data):
    denom = np.max(data,0)-np.min(data,0)
    nume = data-np.min(data,0)
    return nume/denom

# 정규화 되돌리기 함수 
def back_MinMax(data,value):
    diff = np.max(data,0)-np.min(data,0)
    back = value * diff + np.min(data,0)
    return back 

# 데이터 불러오기 
df = pd.read_csv('...\\source.csv')
df = df.dropna()
df['average'] = (df['high'] + df['low'])/2
df = df[::-1] # 역순으로 저장 

df1 = df[['판매량억불','average']].values
df1.shape


# print(df1.shape)

seqLength = 7 # window size 
dataDim = 2 # 입력 데이터 종류
hiddenDim = 10 
outputDim = 1
lr = 0.01
iterations = 500

trainSize = int(len(df1)*0.75)
trainSet = df1[0:trainSize]
testSet = df1[trainSize-seqLength:]

trainSet = MinMaxScaler(trainSet)
testSet = MinMaxScaler(testSet)


# 7일간의 5가지 데이터(시가, 종가, 고가, 저가, 거래량)를 받아와서 
# 바로 다음 날의 종가를 예측하는 모델로 구성

def buildDataSet(timeSeries, seqLength):
    xdata = []
    ydata = [] 
    for i in range(0, len(timeSeries)-seqLength):
        tx = timeSeries[i:i+seqLength,:-1]
        ty = timeSeries[i+seqLength,[-1]]
        xdata.append(tx)
        ydata.append(ty)
    return np.array(xdata), np.array(ydata)
    
trainX, trainY=buildDataSet(trainSet, seqLength)
testX, testY=buildDataSet(testSet, seqLength)

model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(units=8, activation='tanh', input_shape=[7,1]))
model.add(tf.keras.layers.Dense(1))
model.summary()
# 모델 학습과정 설정 
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# 모델 트레이닝 
hist = model.fit(trainX, trainY, epochs=200, batch_size=4)

# 모델 테스트 
res = model.evaluate(testX, testY, batch_size=4)
print("loss",res[0],"mae",res[1])

#7 모델 사용
xhat = testX
yhat = model.predict(xhat)
print(testY)
print(yhat)
print("Evaluate : {}".format(np.average((yhat - testY)**2)))

# 원래 값으로 되돌리기 
predict = back_MinMax(df1[trainSize-seqLength:,[-1]],yhat)
actual = back_MinMax(df1[trainSize-seqLength:,[-1]],testY)
print("예측값",predict)
print("실제값",actual)

plt.figure()
plt.plot(predict[:71], label = "predict_LSTM")
plt.plot(actual[:71],label = "actual")

plt.legend(prop={'size': 15})