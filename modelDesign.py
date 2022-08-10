#=======================================================================================================================
#=======================================================================================================================
"""在modelDesign.py加载自定义的其他模型时，请在modelDesign.py中使用如下代码获取模型路径："""
import os
import sys


import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers


####定义：
def my_upsampling(x, img_w, img_h, method=0):
    """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
    return tf.compat.v1.image.resize_images(x, (img_w, img_h), 2)


def AIModel(x):
    x = layers.Lambda(my_upsampling, arguments={'img_w': 240, 'img_h': 33})(x)

    x1 = layers.Conv2D(filters=16, kernel_size=3,   padding='same')(x)
    x1 = layers.LeakyReLU()(x1)

    num_conv = 24
    num_up = 8
    """下采样环节，深度第一层"""

    conv11 = layers.Conv2D(filters=num_conv, kernel_size=3,   padding='same')(x1) #240*33
    conv11 = layers.LeakyReLU()(conv11)

    pool21 = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(conv11) #120*16
    conv21 = layers.Conv2D(num_conv, 3,   padding='same')(pool21)
    conv21 = layers.LeakyReLU()(conv21)

    pool31 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv21) #60*8
    conv31 = layers.Conv2D(num_conv, 3,   padding='same')(pool31)
    conv31 = layers.LeakyReLU()(conv31)

    pool41 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv31) #30*4
    conv41 = layers.Conv2D(num_conv, 3,   padding='same')(pool41)
    conv41 = layers.LeakyReLU()(conv41)

    pool51 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv41) #15*2
    conv51 = layers.Conv2D(num_conv, 3,   padding='same')(pool51)
    conv51 = layers.LeakyReLU()(conv51)

    #pool61 = layers.MaxPooling2D(pool_size=(3, 1), padding='valid')(conv51)
    #conv61 = layers.Conv2D(num_conv, 3,   padding='same')(pool61)

    #pool71 = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(conv61)
    #conv71 = layers.Conv2D(num_conv, 3,   padding='same')(pool71)



    """深度第二层"""
    #conv72 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(conv71)
    #conv72 = layers.LeakyReLU()(conv72)

    #up72 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 1))(conv72))
    #up72 = layers.LeakyReLU()(up72)
    #merge62 = layers.concatenate([conv61, up72], axis=3)
    #conv62 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge62)
    #conv62 = layers.LeakyReLU()(conv62)

    #up62 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(3, 1))(conv62))
    #up62 = layers.LeakyReLU()(up62)
    #merge52 = layers.concatenate([conv51, up62], axis=3)
    conv52 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(conv51)
    conv52 = layers.LeakyReLU()(conv52)

    up52 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv52))
    up52 = layers.LeakyReLU()(up52)
    merge42 = layers.concatenate([conv41, up52], axis=3)
    conv42 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge42)
    conv42 = layers.LeakyReLU()(conv42)

    up42 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv42))
    up42 = layers.LeakyReLU()(up42)
    merge32 = layers.concatenate([conv31, up42], axis=3)
    conv32 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge32)
    conv32 = layers.LeakyReLU()(conv32)

    up32 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv32))
    up32 = layers.LeakyReLU()(up32)
    merge22 = layers.concatenate([conv21, up32], axis=3)
    conv22 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge22)
    conv22 = layers.LeakyReLU()(conv22)


    up22 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 3))(conv22))
    up22 = layers.LeakyReLU()(up22)
    up22 = layers.Conv2D( num_up, (1, 16),   padding='valid')(up22)
    up22 = layers.LeakyReLU()(up22)
    merge12 = layers.concatenate([x1,conv11, up22], axis=3)
    conv12 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge12)
    conv12 = layers.LeakyReLU()(conv12)

    """深度第三层"""
    #merge73 = layers.concatenate([conv71, conv72], axis=3)
    #conv73 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge73)
    #conv73 = layers.LeakyReLU()(conv73)

    #up73 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 1))(conv73))
    #up73 = layers.LeakyReLU()(up73)
    #merge63 = layers.concatenate([conv61, conv62, up73], axis=3)
    #conv63 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge63)
    #conv63 = layers.LeakyReLU()(conv63)

    #up63 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(3, 1))(conv63))
    #up63 = layers.LeakyReLU()(up63)
    merge53 = layers.concatenate([conv51, conv52], axis=3)
    conv53 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge53)
    conv53 = layers.LeakyReLU()(conv53)

    up53 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv53))
    up53 = layers.LeakyReLU()(up53)
    merge43 = layers.concatenate([conv41, conv42, up53], axis=3)
    conv43 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge43)
    conv43 = layers.LeakyReLU()(conv43)

    up43 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv43))
    up43 = layers.LeakyReLU()(up43)
    merge33 = layers.concatenate([conv31, conv32, up43], axis=3)
    conv33 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge33)
    conv33 = layers.LeakyReLU()(conv33)

    up33 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv33))
    up33 = layers.LeakyReLU()(up33)

    merge23 = layers.concatenate([conv21, conv22, up33], axis=3)
    conv23 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge23)
    conv23 = layers.LeakyReLU()(conv23)



    up23 = layers.Conv2D( num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 3))(conv23))
    up23 = layers.LeakyReLU()(up23)
    up23 = layers.Conv2D( num_up, (1, 16),   padding='valid')(up23)
    up23 = layers.LeakyReLU()(up23)
    merge13 = layers.concatenate([conv11, conv12, up23], axis=3)
    conv13 = layers.Conv2D( num_conv, (3, 3),   padding='SAME')(merge13)
    conv13 = layers.LeakyReLU()(conv13)

    """深度第四层"""

    #merge74 = layers.concatenate([conv71, conv72, conv73], axis=3)
    #conv74 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge74)
    #conv74 = layers.LeakyReLU()(conv74)

    #up74 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 1))(conv74))
    #up74 = layers.LeakyReLU()(up74)
    #merge64 = layers.concatenate([conv61, conv62, conv63, up74], axis=3)
    #conv64 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge64)
    #conv64 = layers.LeakyReLU()(conv64)

    #up64 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(3, 1))(conv64))
    #up64 = layers.LeakyReLU()(up64)
    merge54 = layers.concatenate([conv51, conv52, conv53], axis=3)
    conv54 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge54)
    conv54 = layers.LeakyReLU()(conv54)

    up54 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv54))
    up54 = layers.LeakyReLU()(up54)
    merge44 = layers.concatenate([conv41, conv42, conv43, up54], axis=3)
    conv44 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge44)
    conv44 = layers.LeakyReLU()(conv44)

    up44 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv44))
    up44 = layers.LeakyReLU()(up44)
    merge34 = layers.concatenate([conv31, conv32, conv33, up44], axis=3)
    conv34 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge34)
    conv34 = layers.LeakyReLU()(conv34)

    up34 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv34))
    up34 = layers.LeakyReLU()(up34)

    merge24 = layers.concatenate([conv21, conv22, conv23, up34], axis=3)
    conv24 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge24)
    conv24 = layers.LeakyReLU()(conv24)



    up24 = layers.Conv2D( num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 3))(conv24))
    up24 = layers.LeakyReLU()(up24)
    up24 = layers.Conv2D( num_up, (1, 16),   padding='valid')(up24)
    up24 = layers.LeakyReLU()(up24)
    merge14 = layers.concatenate([x1,conv11, conv12, conv13, up24], axis=3)
    conv14 = layers.Conv2D( num_conv, (3, 3),   padding='SAME')(merge14)
    conv14 = layers.LeakyReLU()(conv14)

    """深度第五层"""
    #merge75 = layers.concatenate([conv71, conv72, conv73, conv74], axis=3)
    #conv75 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge75)
    #conv75 = layers.LeakyReLU()(conv75)

    #up75 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 1))(conv75))
    #up75 = layers.LeakyReLU()(up75)
    #merge65 = layers.concatenate([conv61, conv62, conv63, conv64, up75], axis=3)
    #conv65 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge65)
    #conv65 = layers.LeakyReLU()(conv65)

    #up65 = layers.Conv2D( num_up, 2, padding='same')(layers.UpSampling2D(size=(3, 1))(conv65))
    #up65 = layers.LeakyReLU()(up65)
    merge55 = layers.concatenate([conv51, conv52, conv53, conv54], axis=3)
    conv55 = layers.Conv2D( num_conv, (3, 3),   padding='SAME')(merge55)
    conv55 = layers.LeakyReLU()(conv55)

    up55 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv55))
    up55 = layers.LeakyReLU()(up55)
    merge45 = layers.concatenate([conv41, conv42, conv43, conv44, up55], axis=3)
    conv45 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge45)
    conv45 = layers.LeakyReLU()(conv45)

    up45 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv45))
    up45 = layers.LeakyReLU()(up45)
    merge35 = layers.concatenate([conv31, conv32, conv33, conv34, up45], axis=3)
    conv35 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge35)
    conv35 = layers.LeakyReLU()(conv35)

    up35 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv35))
    up35 = layers.LeakyReLU()(up35)

    merge25 = layers.concatenate([conv21, conv22, conv23, conv24, up35], axis=3)
    conv25 = layers.Conv2D(num_conv, (3, 3), padding='SAME')(merge25)
    conv25 = layers.LeakyReLU()(conv25)



    up25 = layers.Conv2D( num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 3))(conv25))
    up25 = layers.LeakyReLU()(up25)
    up25 = layers.Conv2D( num_up, (1, 16),   padding='valid')(up25)
    up25 = layers.LeakyReLU()(up25)
    merge15 = layers.concatenate([x1,conv11, conv12, conv13, conv14, up25], axis=3)
    conv15 = layers.Conv2D( num_conv, (3, 3),   padding='SAME')(merge15)
    conv15 = layers.LeakyReLU()(conv15)

    """上采样环节"""

    #merge76 = layers.concatenate([conv71, conv72, conv73, conv74, conv75], axis=3)
    #conv76 = layers.Conv2D( num_conv, (3, 3),   padding='SAME')(merge76)
    #conv76 = layers.LeakyReLU()(conv76)

    #up76 = layers.Conv2D( num_up, 2,   padding='same')(layers.UpSampling2D(size=(2, 1))(conv76))
    #up76 = layers.LeakyReLU()(up76)
    #merge66 = layers.concatenate([conv61, conv62, conv63, conv64, conv65, up76], axis=3)
    #conv66 = layers.Conv2D(num_conv, 3,   padding='same')(merge66)
    #conv66 = layers.LeakyReLU()(conv66)

    #up66 = layers.Conv2D( num_up, 2,   padding='same')(layers.UpSampling2D(size=(3, 1))(conv66))
    #up66 = layers.LeakyReLU()(up66)
    merge56 = layers.concatenate([conv51, conv52, conv53, conv54,conv55], axis=3)
    conv56 = layers.Conv2D( num_conv, 3,   padding='same')(merge56)
    conv56 = layers.LeakyReLU()(conv56)

    up56 = layers.Conv2D( num_up, 2,   padding='same')(layers.UpSampling2D(size=(2, 2))(conv56))
    up56 = layers.LeakyReLU()(up56)
    merge46 = layers.concatenate([conv41, conv42, conv43, conv44,conv45 , up56], axis=3)
    conv46 = layers.Conv2D( num_conv, 3,   padding='same')(merge46)
    conv46 = layers.LeakyReLU()(conv46)

    up46 = layers.Conv2D(num_up, 2, padding='same')(layers.UpSampling2D(size=(2, 2))(conv46))
    up46 = layers.LeakyReLU()(up46)
    merge36 = layers.concatenate([conv31, conv32, conv33, conv34, conv35 , up46], axis=3)
    conv36 = layers.Conv2D( num_conv, 3,   padding='same')(merge36)
    conv36 = layers.LeakyReLU()(conv36)

    up36 = layers.Conv2D( num_up, 2,   padding='same')(layers.UpSampling2D(size=(2, 2))(conv36))
    up36 = layers.LeakyReLU()(up36)

    merge26 = layers.concatenate([conv21, conv22, conv23, conv24,conv25 , up36], axis=3)
    conv26 = layers.Conv2D(num_conv, 3,   padding='same')(merge26)
    conv26 = layers.LeakyReLU()(conv26)

    up26 = layers.Conv2D( num_up, 2,   padding='same')(layers.UpSampling2D(size=(2, 3))(conv26))
    up26 = layers.LeakyReLU()(up26)
    up26 = layers.Conv2D( num_up, (1, 16),   padding='valid')(up26)
    up26 = layers.LeakyReLU()(up26)
    merge16 = layers.concatenate([x1,conv11, conv12, conv13, conv14, conv15 ,up26 ], axis=3)
    conv16 = layers.Conv2D(num_conv, 3,   padding='same')(merge16)
    conv16 = layers.LeakyReLU()(conv16)

    #####输出
    conv_out = layers.Conv2D(2, 3, padding='same')(conv16)
    conv_out = layers.LeakyReLU()(conv_out)

    output = conv_out

    return output


def NMSE(x, x_hat):
    x_real = x[:, :, :, 0]
    x_imag = x[:, :, :, 1]
    x_hat_real = x_hat[:, :, :, 0]
    x_hat_imag = x_hat[:, :, :, 1]

    #x_C = x_real + 1j * (x_imag)
    #x_hat_C = x_hat_real + 1j * (x_hat_imag)
    power = K.sum(x_real ** 2 + x_imag ** 2)
    mse = K.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) **2)
    nmse = mse / power
    return nmse
