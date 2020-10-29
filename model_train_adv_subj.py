# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, auc, accuracy_score, classification_report
from tensorflow.keras import regularizers, layers, optimizers, Model, Input
from os.path import join

scaler = StandardScaler()

early_stop_clbk = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
nhu = 2048
nhl = 5
nClasses_att = 2
nClasses_age = 6
seed = 44
np.random.seed(seed)

#Design gradient scale layer
@tf.custom_gradient
def grad_scale(x,scale=1.0):
    y = tf.identity(x)
    def custom_grad(dy):
        return [scale*dy, None]
    return y, custom_grad

class GradScale(layers.Layer):
    def __init__(self, scale=1.0, name='GRU'):
        super().__init__(name=name)
        self.scale = scale
        #self.name = name

    def call(self, x):
        return grad_scale(x,self.scale)


#Adv training parameters
lNGenLayers = [3]
lNODLayers = [2]
GRU_Scale = [-0.001]
lNEpochs = [2]#,3,4,5]
lNUnits = [2048]
Trials = np.array(np.meshgrid(lNGenLayers,lNODLayers,lNUnits,GRU_Scale,lNEpochs)).T.reshape(-1,5)
sPath = 'models'
sPrefix = 'Adv_var'
loss={'att': 'sparse_categorical_crossentropy', 'age': 'sparse_categorical_crossentropy'}
optimizer = tfa.optimizers.AdamW(weight_decay=0.001, learning_rate=0.0001)
fSum = open('model_sum_Adv_0.0001_Var','w')


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

nDim = X_train.shape[1]

print('Prepare age data...')

age_train = data_train['age']
age_valid = data_valid['age']
age_test = data_test['age']

samples_test_age = np.where(age_test == 0)[0].shape[0]
samples_valid_age = np.where(age_valid == 0)[0].shape[0]
samples_train_age = np.where(age_train == 0)[0].shape[0]

index_test_age = np.where(age_test==0)[0]
index_valid_age = np.where(age_valid==0)[0]
index_train_age = np.where(age_train==0)[0]


for i in [1,2,3,4,5]:
    index_test_age_tmp = np.where(age_test==i)[0]
    index_valid_age_tmp = np.where(age_valid==i)[0]
    index_train_age_tmp = np.where(age_train==i)[0]

    np.random.shuffle(index_test_age_tmp)
    np.random.shuffle(index_valid_age_tmp)
    np.random.shuffle(index_train_age_tmp)


    index_test_age = np.r_[index_test_age,index_test_age_tmp[:samples_test_age]]
    index_valid_age = np.r_[index_valid_age,index_valid_age_tmp[:samples_valid_age]]
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


for att in range(y_train.shape[1]):
    index_train_p = np.where(y_train[:,att]==1)[0]
    index_train_n = np.where(y_train[:,att]==0)[0]
    index_valid_p = np.where(y_valid[:,att]==1)[0]
    index_valid_n = np.where(y_valid[:,att]==0)[0]
    index_test_p = np.where(y_test[:,att]==1)[0]
    index_test_n = np.where(y_test[:,att]==0)[0]

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

    y_train_att = y_train[index_train][:,att]
    y_valid_att = y_valid[index_valid][:,att]
    y_test_att = y_test[index_test][:,att]
    
    y_train_att_age = age_train[index_train]
    y_valid_att_age = age_valid[index_valid]
    y_test_att_age = age_test[index_test]

    y_train_age_att = y_train[index_train_age][:,att]
    y_valid_age_att = y_valid[index_valid_age][:,att]
    y_test_age_att = y_test[index_test_age][:,att]
 

    X_train_att = X_train[index_train]
    X_valid_att = X_valid[index_valid]
    X_test_att = X_test[index_test]


    scaler.fit(X_train_att)
    X_train_att_std = scaler.transform(X_train_att)
    X_valid_att_std = scaler.transform(X_valid_att)
    X_test_att_std = scaler.transform(X_test_att)

    print('Start training attribute {0} with {1} training samples, {2} validation samples, {3} testing samples'.format(att,y_train_att.shape[0],y_valid_att.shape[0],y_test_att.shape[0]), flush=True)
    print('Adversarial Training to age with {0} training samples, {1} validation samples, {2} testing samples'.format(y_train_age.shape[0],y_valid_age.shape[0],y_test_age.shape[0]), flush=True)

    for nGenLayers, nDOLayers, nUnits, GRU_Scale, nEpochs in Trials:
        nGenLayers = int(nGenLayers)
        nOutLayers = int(nDOLayers)
        nAgeLayers = int(nDOLayers)
        nRepeats = 5

        GRU_scale = GRU_Scale
        nUnits = int(nUnits)
        nEpochs = int(nEpochs)
        bDropout = False

        name = '_'.join([str(s) for s in (att,nGenLayers, nDOLayers, nUnits, GRU_Scale, nEpochs)])
        
        #Start Model building
        
        Input_Layer = Input(shape=nDim, name='Input')
        prev_layer = Input_Layer

        for i in range(nGenLayers):
            sLayerName = 'G_'+str(i)
            x = layers.Dense(nUnits, activation='relu', name=sLayerName)(prev_layer)
            if bDropout:
                x = layers.Dropout(0.5)(x)
            
            prev_layer = x            

        split_layer = x

        for i in range(nOutLayers):
            sLayerName = 'O_'+str(i)
            x = layers.Dense(nUnits, activation='relu', name=sLayerName)(prev_layer)
            prev_layer = x

        last_out_layer = x

        prev_layer = split_layer

        #Add Gradient reverse layer
        Grad_layer = GradScale(scale=GRU_scale)(prev_layer)

        prev_layer = Grad_layer
        for i in range(nAgeLayers):
            sLayerName = 'D_'+str(i)
            x = layers.Dense(nUnits, activation='relu', name=sLayerName)(prev_layer)
            prev_layer = x

        last_Age_layer = x


        #Adding output layers
        output1 = layers.Dense(units=nClasses_att,activation='softmax', name='att')(last_out_layer)
        output2 = layers.Dense(units=nClasses_age,activation='softmax', name='age')(last_Age_layer)
       
        model = Model(inputs=[Input_Layer], outputs=[output1, output2])
        model.compile(optimizer=optimizer,loss=loss, metrics=['accuracy'])
        #keras.utils.plot_model(model,join(sPath,''.join([sPrefix,name,'.png'])))



        #Do Training
        
        G_Final_Model = None
        for r in range(nRepeats):
          print('Train G+O....',flush=True)
          GRU_scale_crnt = GRU_scale * (r+1)
          #First Train G & O for 2 epochs
          for i in range(nAgeLayers):
            sLayerName = 'D_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = False
          for i in range(nGenLayers):
            sLayerName = 'G_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = True
          for i in range(nOutLayers):
            sLayerName = 'O_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = True
          x = model.get_layer(name='GRU')
          x.scale = 0.0
          model.fit(X_train_att_std, {'att':y_train_att, 'age':y_train_att_age}, epochs=nEpochs, batch_size=128, validation_data=(X_valid_att_std, {'att':y_valid_att,'age':y_valid_att_age}))
          model.save_weights(join(sPath,''.join([sPrefix,name,'_',str(r),'.wigts'])))
          y_p_valid = np.argmax(model.predict(X_valid_att_std)[0],axis=1)
          y_p_test = np.argmax(model.predict(X_test_att_std)[0],axis=1)
          with open(join(sPath,''.join([sPrefix,name,'_',str(r),'.res'])),'w') as fRes:
            test_acc = accuracy_score(y_test_att,y_p_test)
            valid_acc = accuracy_score(y_valid_att,y_p_valid)
            print('Test_results****************', file = fRes, flush=True)
            print(test_acc, file = fRes, flush=True)
            print(confusion_matrix(y_test_att,y_p_test), file = fRes,flush=True)
            print(classification_report(y_test_att,y_p_test), file = fRes,flush=True)
            print('Valid_results****************', file = fRes,flush=True)
            print(valid_acc, file = fRes,flush=True)
            print(confusion_matrix(y_valid_att,y_p_valid), file = fRes,flush=True)
            print(classification_report(y_valid_att,y_p_valid), file = fRes,flush=True)
          print('Att {} R {} Model {} test accuracy = {} valid accuracy = {}'.format(att,r,name,test_acc, valid_acc), file = fSum, flush=True)
          fSum.flush()
          #G_Final_Model = copy(model)
          #Second Train D
          print('Train D....',flush=True)
          for i in range(nAgeLayers):
            sLayerName = 'D_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = True
          for i in range(nGenLayers):
            sLayerName = 'G_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = False
          for i in range(nOutLayers):
            sLayerName = 'O_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = False
          x = model.get_layer(name='GRU')
          x.scale = 1.0
      
          model.fit(X_train_age_std, {'att':y_train_age_att, 'age':y_train_age}, epochs=nEpochs, batch_size=128, validation_data=(X_valid_age_std, {'att':y_valid_age_att,'age':y_valid_age}))
      
      
          #Finally G Adv to D
          print('Train G Adv to D....',flush=True)
          for i in range(nAgeLayers):
            sLayerName = 'D_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = False
          for i in range(nGenLayers):
            sLayerName = 'G_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = True
          for i in range(nOutLayers):
            sLayerName = 'O_'+str(i)
            x = model.get_layer(name=sLayerName)
            x.trainable = False
          x = model.get_layer(name='GRU')
          x.scale = GRU_scale_crnt
      
          model.fit(X_train_age_std, {'att':y_train_age_att, 'age':y_train_age}, epochs=nEpochs, batch_size=128, validation_data=(X_valid_age_std, {'att':y_valid_age_att,'age':y_valid_age}))

fSum.close()


