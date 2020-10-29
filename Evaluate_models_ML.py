import numpy as np
import pandas as pd
from os.path import join,isdir
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, auc, accuracy_score, classification_report
from tensorflow.keras import regularizers, layers, optimizers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


scaler = StandardScaler()
data_train = np.load('TrainData.npz')
data_valid = np.load('ValidData.npz')
data_test = np.load('TestData.npz')

print('Loading data....', flush=True)
X_train = data_train['X']
X_valid = data_valid['X']
X_test = data_test['X']
y_train = data_train['y']
y_valid = data_valid['y']
y_test = data_test['y']

mask = np.ones(25,dtype=bool)

for i in (2,5,6,23,24):
    mask[i] = False

y_train = y_train[:,mask]
y_test = y_test[:,mask]
y_valid = y_valid[:,mask]

scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
X_valid_std = scaler.transform(X_valid)

model = tf.keras.models.load_model('ML_model.h5')
y_p_v = model.predict(X_valid_std)
y_p_v_i = (y_p_v > 0.5).astype(int)

y_p_t = model.predict(X_test_std)
y_p_t_i = (y_p_t > 0.5).astype(int)


for i in range(20):
    if i in [0,1]:
        j = i
    elif i in [2,3]:
        j = i+1
    else:
        j = i + 3

    f = open('_'.join([str(j),'class_report_ML']),'w')
    val_bal_ac = balanced_accuracy_score(y_valid[:,i],y_p_v_i[:,i])
    print(classification_report(y_valid[:,i],y_p_v_i[:,i]),file=f)
    y_p = np.argmax(model.predict(X_test_std),axis=1)
    test_bal_ac = balanced_accuracy_score(y_test[:,i],y_p_t_i[:,i])
    print(classification_report(y_test[:,i],y_p_t_i[:,i]),file=f)
    print('att {} test acc {} valid acc {}'.format(j,test_bal_ac,val_bal_ac), file=f)
f.close()    
