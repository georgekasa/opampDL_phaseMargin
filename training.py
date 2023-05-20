import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
# tf.keras.layers.Flatten(
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
import numba as nb
from tensorflow import keras



def plotTrainingLoss(theHistory):
    plt.semilogy(theHistory.history['loss'])
    plt.semilogy(theHistory.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()


@nb.jit(nopython=True, parallel=True)
def nmeanstd(a, n):
    b = []; c = []
    for i in range(n):
        b.append(np.mean(a))
        c.append(np.std(a))
    
    return b, c


data_dir = r'/home/gkasap/Documents/Python/opampDL/pmosOpAmp/'

df_path = os.path.join(data_dir, 'filterData.pkl')
df = pd.read_pickle(df_path)
# for key in df.keys():
#     print(key)
#     print("################")

df = df.astype('float32')
df["CapMiller"] = df["CapMiller"] * 1e12
train, test = train_test_split(df, test_size=0.2, random_state=84)
listTrain = ["Mdiff", "M_out", "ResistorMiller", "CapMiller"]
x_train, x_test, y_train, y_test = train[listTrain], test[listTrain], train['phaseMargin'], test["phaseMargin"]

start_time = time.time()
#xtrain_mean, xtrain_std = np.mean(x_train, axis = 0), np.std(x_train, axis = 0)
xtrain_mean, xtrain_std = np.mean(x_train, axis = 0), np.std(x_train, axis = 0)
#xtrain_mean, xtrain_std = nmeanstd(x_train['M_out'].values, len(x_train['M_out']))
print("--- %s seconds ---" % (time.time() - start_time))
xtrain_norm = (x_train - xtrain_mean) / (xtrain_std + 0.000001)
xtest_norm = (x_test - xtrain_mean) / (xtrain_std + 0.000001)
y_train = y_train.to_frame()
y_test = y_test.to_frame()
print(xtest_norm.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=[4], kernel_initializer = 'he_normal'))
#model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer = 'he_normal'))
#model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer = 'he_normal'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='relu', bias_initializer=tf.keras.initializers.Constant(np.mean(y_train))))


OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=OPTIMIZER,
              metrics =['MeanSquaredError'])
model.summary()
EPOCHS = 400
BATCH_SIZE = 64
history = model.fit(xtrain_norm, y_train, validation_data=(
    xtest_norm, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE,
    verbose=2, shuffle=True)

#history.best = np.min(history.history['val_mean_absolute_error'])

#Print first 4 predictions.
predictions = model.predict(xtest_norm)
for i in range(0, 10):
  print('Prediction: ', predictions[i, 0],' true value: ', y_test.iloc[i])

plotTrainingLoss(history)




#1) need more data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 648 all the dataset




