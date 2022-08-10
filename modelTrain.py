import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' #å±è½log
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #å¼ºå¶GPUè¿è¡ä»£ç ,7ä»£è¡¨ç¬?ä¸ªGPU

import numpy as np
import h5py
import tensorflow as tf
import keras
import time

from SRCNN_DnCNN import AIModel,NMSE
from scipy import io as scio
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
#import matplotlib.pyplot as plt
from keras.models import load_model
from keras_flops import get_flops

# Data loading
Input_name = 'H_in_120km_train.mat'
Output_name = 'H_out_120km_train.mat'

print('The current dataset is : %s'%(Input_name))
H_in = h5py.File(Input_name,'r')

X = np.transpose(H_in['H_in'][:])
print('X',X.shape)

print('The current dataset is : %s'%(Output_name))
H_out = h5py.File(Output_name,'r')

Y = np.transpose(H_out['H_out'][:])
print('Y',Y.shape)

# model
Model_input1 = keras.Input(shape=(240, 5, 2), name="Input_Channel")
"""
低版本int不能当shape的值，应该写成shape=(1,)
"""
#Model_input2 = keras.Input(shape=(1,), name="SNR")

Model_output = AIModel(Model_input1)


AI_Model = keras.Model(inputs=Model_input1, outputs=Model_output, name='AI_Model')
AI_Model.compile(optimizer='adam', loss='mse')
print(AI_Model.summary())
flops = get_flops(AI_Model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")

# model training
#tensorboard = TensorBoard(log_dir='SRCNN_DnCNN')
#checkpoint = ModelCheckpoint(filepath='SRCNN_DnCNN.h5',monitor='val_loss',mode='min' ,save_best_only='True')
#callback_lists=[tensorboard,checkpoint]
#history = AI_Model.fit(x=[X], y=Y,batch_size=128,epochs=1000, verbose=1, shuffle=1,validation_split=0.05, validation_data=(),callbacks=callback_lists)

# model save
#AI_Model.save('CNNBilstm.h5')

# 预测


AI_Model = load_model('SRCNN_DnCNN.h5', custom_objects={'tf': tf})
tic = time.time()
preds = AI_Model.predict(X)
toc = time.time()
shijian = toc-tic
print(shijian)
scio.savemat('SRCNN_DnCNN.mat', {'preds':preds})  # 把data存到data_mat里


#scio.savemat('train_loss.mat', {'loss':history.history['loss']})
#scio.savemat('test_loss.mat', {'loss':history.history['val_loss']})
print('END')
