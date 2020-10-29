# coding: utf-8
import numpy as np
import pandas as pd
from os.path import join
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, auc, accuracy_score, classification_report
from tensorflow.keras import regularizers, layers, optimizers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

scaler = StandardScaler()

early_stop_clbk = EarlyStopping(monitor='val_loss', mode='min', patience=10)

bDelta = False
nContextFrames = 5
nFeatures = 26
EgsVecSize = nContextFrames*nFeatures*3
mask_noDelta = np.zeros(EgsVecSize,dtype=bool)
if bDelta:
    mask = ~mask_noDelta
else:
    for i in range(nContextFrames):
      mask_noDelta[i*nFeatures*3:i*nFeatures*3+nFeatures] = True
    mask = mask_noDelta


nhu = 2048
nhl = 5
lr=0.0001
nClasses = 2
epochs = 30

MAX_TRAIN = 9000000
MAX_VALID = 900000
MAX_TEST = 900000

affix = '_'.join(['Model',str(nhu),str(nhl),str(lr),'noDelta'])
fRes = open('_'.join(['Results',affix]),'w')

data_train = np.load('TrainData.npz')
data_valid = np.load('ValidData.npz')
data_test = np.load('TestData.npz')

print('Loading data....', flush=True)
X_train = data_train['X'][:,mask]
X_valid = data_valid['X'][:,mask]
X_test = data_test['X'][:,mask]
y_train = data_train['y']
y_valid = data_valid['y']
y_test = data_test['y']

for att in range(y_train.shape[1]):
    index_train_p = np.where(y_train[:,att]==1)[0]
    index_train_n = np.where(y_train[:,att]==0)[0]
    index_valid_p = np.where(y_valid[:,att]==1)[0]
    index_valid_n = np.where(y_valid[:,att]==0)[0]
    index_test_p = np.where(y_test[:,att]==1)[0]
    index_test_n = np.where(y_test[:,att]==0)[0]

    samples_train = min(MAX_TRAIN,index_train_p.shape[0])
    samples_valid = min(MAX_VALID,index_valid_p.shape[0])
    samples_test = min(MAX_TEST,index_test_p.shape[0])

    if samples_train == 0:
        continue

    np.random.shuffle(index_train_n)
    np.random.shuffle(index_train_p)
    np.random.shuffle(index_valid_p)
    np.random.shuffle(index_valid_n)
    np.random.shuffle(index_test_n)
    np.random.shuffle(index_test_p)

    index_train_n = index_train_n[:samples_train]
    index_train_p = index_train_p[:samples_train]
    index_valid_n = index_valid_n[:samples_valid]
    index_valid_p = index_valid_p[:samples_valid]
    index_test_p = index_test_p[:samples_test]
    index_test_n = index_test_n[:samples_test]
    
    index_train = np.r_[index_train_p,index_train_n]
    index_valid = np.r_[index_valid_p,index_valid_n]
    index_test = np.r_[index_test_p,index_test_n]

    y_train_att = y_train[index_train][:,att]
    y_valid_att = y_valid[index_valid][:,att]
    y_test_att = y_test[index_test][:,att]


    X_train_att = X_train[index_train]
    X_valid_att = X_valid[index_valid]
    X_test_att = X_test[index_test]


    scaler.fit(X_train_att)
    X_train_std = scaler.transform(X_train_att)
    X_valid_std = scaler.transform(X_valid_att)
    X_test_std = scaler.transform(X_test_att)

    print('Start training attribute {0} with {1} training samples, {2} validation samples, {3} testing samples'.format(att,y_train_att.shape[0],y_valid_att.shape[0],y_test_att.shape[0]), flush=True)
    mc = ModelCheckpoint(join('models_att_train','_'.join([str(att),'best_model.h5',affix])), monitor='val_loss', mode='min', save_best_only=True)
    
    nDim = X_train_std.shape[1]
    input_layer = Input(shape=nDim)
    x = layers.Dense(units=nhu,activation=tf.nn.relu)(input_layer)
    for i in range(nhl-1):
        x = layers.Dense(units=nhu,activation=tf.nn.relu)(x)
    output_layer = layers.Dense(units=nClasses,activation=tf.nn.softmax)(x)
    
    model = Model(inputs=input_layer,outputs=output_layer)
    model.compile(optimizer=tfa.optimizers.AdamW(weight_decay=0.001, learning_rate=lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train_std, y_train_att ,epochs=epochs,batch_size=128, validation_data=(X_valid_std, y_valid_att),callbacks=[early_stop_clbk,mc])

    print('Model Evaluation....', flush=True)
    print('Validation set', flush=True)
    y_p = np.argmax(model.predict(X_valid_std),axis=1)
    valid_acc = accuracy_score(y_valid_att,y_p) 
    print(classification_report(y_valid_att,y_p), flush=True)
    print('Test set', flush=True)
    y_p = np.argmax(model.predict(X_test_std),axis=1)
    test_acc = accuracy_score(y_test_att,y_p)
    print(classification_report(y_test_att,y_p), flush=True)
    print('Att {} test_acc {} valid_acc {}'.format(att,test_acc,valid_acc),file=fRes, flush=True)
fRes.close()
