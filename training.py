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

def plotTraining(theHistory):
    plt.plot(theHistory.history['mean_absolute_error'])
    plt.plot(theHistory.history['val_mean_absolute_error'])
    plt.title('Model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()

def plotTrainingLoss(theHistory):
    plt.plot(theHistory.history['loss'])
    plt.plot(theHistory.history['val_loss'])
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
train, test = train_test_split(df, test_size=0.2, random_state=84)
listTrain = ["Mdiff", "M_out", "ResistorMiller", "CapMiller"]
x_train, x_test, y_train, y_test = train[listTrain], test[listTrain], train['phaseMargin'], test["phaseMargin"]

start_time = time.time()
#xtrain_mean, xtrain_std = np.mean(x_train, axis = 0), np.std(x_train, axis = 0)
xtrain_mean, xtrain_std = np.mean(x_train['M_out'].values), np.std(x_train['M_out'].values)
#xtrain_mean, xtrain_std = nmeanstd(x_train['M_out'].values, len(x_train['M_out']))
print("--- %s seconds ---" % (time.time() - start_time))
xtrain_norm = (x_train - xtrain_mean) / (xtrain_std + 0.000001)
xtest_norm = (x_test - xtrain_mean) / (xtrain_std + 0.000001)
y_train = y_train.to_frame()
y_test = y_test.to_frame()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=[4]))
#model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam',
              metrics =['mean_absolute_error'])
model.summary()
EPOCHS = 400
BATCH_SIZE = 32
history = model.fit(xtrain_norm, y_train, validation_data=(
    xtest_norm, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE,
    verbose=2, shuffle=True)



#Print first 4 predictions.
predictions = model.predict(xtest_norm)
for i in range(0, 10):
  print('Prediction: ', predictions[i, 0],' true value: ', y_test.iloc[i])

plotTrainingLoss(history)




########### many things to improve the code###################
#1) ADD NORMALIZATION
#2) change leaky RELU 
#3) need more data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!







"""
17/17 - 0s - loss: 17.2926 - mean_absolute_error: 3.2753 - val_loss: 13.6213 - val_mean_absolute_error: 3.0206 - 35ms/epoch - 2ms/step
5/5 [==============================] - 0s 535us/step
Prediction:  69.98892  true value:  phaseMargin    64.162903
Name: 411, dtype: float32
Prediction:  64.20131  true value:  phaseMargin    64.819298
Name: 666, dtype: float32
Prediction:  71.89119  true value:  phaseMargin    67.406097
Name: 417, dtype: float32
Prediction:  68.13321  true value:  phaseMargin    69.774002
Name: 1035, dtype: float32
Prediction:  71.20552  true value:  phaseMargin    66.153198
Name: 1431, dtype: float32
Prediction:  76.62684  true value:  phaseMargin    66.522301
Name: 262, dtype: float32
Prediction:  72.42052  true value:  phaseMargin    75.875198
Name: 613, dtype: float32
Prediction:  68.98981  true value:  phaseMargin    66.206596
Name: 1572, dtype: float32
Prediction:  70.880005  true value:  phaseMargin    77.155296
Name: 756, dtype: float32
Prediction:  68.13784  true value:  phaseMargin    65.222198

"""
















    # xtrain_norm = xtrain_norm.to_numpy()
# xtest_norm = xtest_norm.to_numpy()
# y_train = y_train.to_numpy()
# y_test = y_test.to_numpy()


#X_train, X_test, y_train, y_test = train_test_split(valid_df[["Mdiff", "M_out", "ResistorMiller", "CapMiller"]], valid_df["phaseMargin"], test_size=0.2, random_state=42)
#df[(df.Mdiff == 8) & (df.M_out == 32) & (df.ResistorMiller == 50)]
####Normalization############

# x = valid_df.to_numpy()

# # Compute the mean and standard deviation of the training data
# xtrain_mean, xtrain_std = np.mean(x_train), np.std(x_train)

# # Normalize the training data
# xtrain_norm = (x_train - xtrain_mean) / xtrain_std

# # Normalize the test data using the training data's mean and standard deviation
# xtest_norm = (x_train - xtrain_mean) / xtrain_std

#############################

#valid_df = df.drop(columns = ['Ldiff',"Lout", "LmirOut", "LactiveLoad", "LmirOut", "Lout", "Lmirdiff", "Wdiff", "WactiveLoad", "Wout", "WmirOut", "Wmirdiff", "Mmirdiff", "MmirOut", "MactiveLoad"])
#x = tf.convert_to_tensor(df[["Mdiff", "M_out", "ResistorMiller", "CapMiller"]])
#y = tf.convert_to_tensor(df["phaseMargin"])

# Split the data into training and test sets