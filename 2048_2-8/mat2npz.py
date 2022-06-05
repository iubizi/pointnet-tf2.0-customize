####################
# 数据
####################

import scipy.io as scio
import numpy as np

data = scio.loadmat('adc_data1_allframe_2048.mat') \
       ['xyzdist_2048total']

for i in range(2, 9):
    temp = scio.loadmat('adc_data'+str(i)+'_allframe_2048.mat') \
           ['xyzdist_2048total']
    data = np.concatenate( (data, temp) )

print(data.shape)
data = data.transpose(0, 2, 1)
print(data.shape)
print('='*20)
print(data)
print()

####################
# 标签
####################

label = [0]*int(data.shape[0]>>1) + [1]*int(data.shape[0]>>1)
label = np.array(label)

from tensorflow.keras.utils import to_categorical
label = to_categorical(label)

print(label.shape)
print('='*20)
print(label)
print()

####################
# 拆分
####################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    data, label, test_size=0.33, random_state=42, stratify=label)

####################
# 可视化
####################

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)
print()

####################
# 存储为npz压缩格式
####################

np.savez_compressed( 'customize.npz',
          
                     x_train = x_train,
                     y_train = y_train,
          
                     x_test = x_test,
                     y_test = y_test )
