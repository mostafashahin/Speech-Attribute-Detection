# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, auc, accuracy_score, classification_report
from tensorflow.keras import regularizers, layers, optimizers, Model, Input


scaler = StandardScaler()

early_stop_clbk = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
nhu = 1024
nhl = 5
nClasses = 2


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

age_train = data_train['age']
age_valid = data_valid['age']
age_test = data_test['age']

for age in [3,4,5]:
    index_train = np.where(age_train==age)[0]
    index_test = np.where(age_test==age)[0]
    index_valid = np.where(age_valid==age)[0]

    X_train_age = X_train[index_train]
    X_valid_age = X_valid[index_valid]
    X_test_age = X_test[index_test]

    y_train_age = y_train[index_train]
    y_valid_age = y_valid[index_valid]
    y_test_age = y_test[index_test]

    for att in range(y_train.shape[1]):
        index_train_p = np.where(y_train_age[:,att]==1)[0]
        index_train_n = np.where(y_train_age[:,att]==0)[0]
        index_valid_p = np.where(y_valid_age[:,att]==1)[0]
        index_valid_n = np.where(y_valid_age[:,att]==0)[0]
        index_test_p = np.where(y_test_age[:,att]==1)[0]
        index_test_n = np.where(y_test_age[:,att]==0)[0]
    
        samples_train = min(1000000,index_train_p.shape[0])
        samples_valid = min(100000,index_valid_p.shape[0])
        samples_test = min(100000,index_test_p.shape[0])
    
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
    
        y_train_att = y_train_age[index_train][:,att]
        y_valid_att = y_valid_age[index_valid][:,att]
        y_test_att = y_test_age[index_test][:,att]
    
    
        X_train_att = X_train_age[index_train]
        X_valid_att = X_valid_age[index_valid]
        X_test_att = X_test_age[index_test]
    
    
        scaler.fit(X_train_att)
        X_train_std = scaler.transform(X_train_att)
        X_valid_std = scaler.transform(X_valid_att)
        X_test_std = scaler.transform(X_test_att)
    
        print('Start training age {0} attribute {1} with {2} training samples, {3} validation samples, {4} testing samples'.format(age,att,y_train_att.shape[0],y_valid_att.shape[0],y_test_att.shape[0]), flush=True)
        nDim = X_train_std.shape[1]
        input_layer = Input(shape=nDim)
        x = layers.Dense(units=nhu,activation=tf.nn.relu)(input_layer)
        for i in range(nhl-1):
            x = layers.Dense(units=nhu,activation=tf.nn.relu)(x)
        output_layer = layers.Dense(units=nClasses,activation=tf.nn.softmax)(x)
        
        model = Model(inputs=input_layer,outputs=output_layer)
        model.compile(optimizer=tfa.optimizers.AdamW(weight_decay=0.001, learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(X_train_std, y_train_att ,epochs=20,batch_size=128, validation_data=(X_valid_std, y_valid_att),callbacks=[early_stop_clbk])
    
        print('Model Evaluation....', flush=True)
        print('Validation set', flush=True)
        y_p = np.argmax(model.predict(X_valid_std),axis=1)
        print(classification_report(y_valid_att,y_p), flush=True)
        print('Test set', flush=True)
        y_p = np.argmax(model.predict(X_test_std),axis=1)
        print(classification_report(y_test_att,y_p), flush=True)

