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
nhu = 2048
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

print('Test data 0: {0} 1: {1} 2: {2} 3: {3} 4: {4} 5: {5}'.format(*np.bincount(age_test)))
print('Valid data 0: {0} 1: {1} 2: {2} 3: {3} 4: {4} 5: {5}'.format(*np.bincount(age_valid)))
print('Train data 0: {0} 1: {1} 2: {2} 3: {3} 4: {4} 5: {5}'.format(*np.bincount(age_train)))
samples_test_age = np.where(age_test == 0)[0].shape[0]
samples_valid_age = np.where(age_valid == 0)[0].shape[0]
samples_train_age = np.where(age_train == 0)[0].shape[0]

index_test_age = np.where(age_test==0)[0]

for i in [1,2,3,4,5]:
    index_test_age_tmp = np.where(age_test==i)[0]
    np.random.shuffle(index_test_age_tmp)
    index_test_age = np.r_[index_test_age,index_test_age_tmp[:samples_test_age]]
index_valid_age = np.where(age_valid==0)[0]

for i in [1,2,3,4,5]:
    index_valid_age_tmp = np.where(age_valid==i)[0]
    np.random.shuffle(index_valid_age_tmp)
    index_valid_age = np.r_[index_valid_age,index_valid_age_tmp[:samples_valid_age]]
index_train_age = np.where(age_train==0)[0]

for i in [1,2,3,4,5]:
    index_train_age_tmp = np.where(age_train==i)[0]
    np.random.shuffle(index_train_age_tmp)
    index_train_age = np.r_[index_train_age,index_train_age_tmp[:samples_train_age]]

X_train_age = X_train[index_train_age]
X_test_age = X_test[index_test_age]
X_valid_age = X_valid[index_valid_age]
y_train_age = age_train[index_train_age]
y_test_age = age_test[index_test_age]
y_valid_age = age_valid[index_valid_age]

scaler.fit(X_train_age)
X_train_age_std = scaler.transform(X_train_age)
X_valid_age_std = scaler.transform(X_valid_age)
X_test_age_std = scaler.transform(X_test_age)


nDim = X_train_age_std.shape[1]
input_layer = Input(shape=nDim)
x = layers.Dense(units=nhu,activation=tf.nn.relu)(input_layer)
for i in range(nhl-1):
    x = layers.Dense(units=nhu,activation=tf.nn.relu)(x)
output_layer = layers.Dense(units=nClasses,activation=tf.nn.softmax)(x)
nClasses
model = Model(inputs=input_layer,outputs=output_layer)
model.compile(optimizer=tfa.optimizers.AdamW(weight_decay=0.001, learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train_age_std, y_train_age ,epochs=20,batch_size=128, validation_data=(X_valid_age_std, y_valid_age),callbacks=[early_stop_clbk])


y_p = model.predict(X_valid_age_std)
y_p_i = np.argmax(y_p,axis=1)
print(classification_report(y_valid_age,y_p_i))
y_p = model.predict(X_test_age_std)
y_p_i = np.argmax(y_p,axis=1)
print(classification_report(y_test_age,y_p_i))

