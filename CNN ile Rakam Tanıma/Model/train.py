from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.utils import to_categorical, plot_model
from keras. layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np

import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
filterwarnings('ignore')

# Mnist veri seti keras küütphanesi içerisinde tanımlı olan veri setlerinden birisidir.
# Mnist veri setinin yüklenmesi
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Eğitim Seti Boyutu:", x_train.shape, y_train.shape ,"Test Seti Boyutu:", x_test.shape, y_test.shape)

# Bağımlı değişkendeki etiket sayısının hesaplanması. Burada 0-9 arasında rakamlar olduğu için 10 sınıf bulunmaktadır.
num_labels = len(np.unique(y_train))
print("Sınıf Sayısı:" ,num_labels)

plt.figure(figsize=(14,14))
x, y = 4,3
for i in range(12):
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i])
plt.show()

# One Hot Encoding İşlemi
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Girdi görüntüsünün boyutları
image_size = x_train.shape[1]

# reshape işlemi gerçekleştirilmesi
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])

# Ayrıca 0 ile 255 arasındaki gri skaladaki piksel değerlerini 0 ile 1 arasında normalize edeceğiz. Bunu her birini 255'e bölerek yapıyoruz.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Hiperparamtrelerin Ayarlanması

# Giriş görüntüsünün boyutları (28, 28, 1) belirtilmektedir.
input_shape = (image_size, image_size, 1)

# Her bir epochda (dönemde) veri setinden alınacak olan kümenin eleman sayısı belirlenir.
batch_size = 128

# Bağımlı değişkendeki etiket sayısının hesaplanması. Burada 0-9 arasında rakamlar olduğu için 10 sınıf bulunmaktadır.
# num_labels = len(np.unique(y_train))

# Modelin kaç dönemde (epochs) eğitilmesi gerektiği ile ilgili hiperparametredir.
epochs = 10 # 12 epoch önerilir

# Filtre Boyutu 3 x 3 olacak şekilde ayarlanmıştır.
kernel_size = 3

# Ortaklama (Pooling) Boyutu 2 x 2 olacak şekilde ayarlanmıştır.
pool_size = 2

# Kaç adet filtre olacağı bilgisi ayarlanmıştır. 64 adet filtre olacaktır.
filters = 64

# Seyreltme oranı ayarlanmıştır.
dropout = 0.2

# Model Katmanlarının Oluşturulması 9 Katmandan oluşmaktadır.
model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())

model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

# Değerlendirme metriği olarak Accuracy kullanılmaktadır. Bunun nedeni Sınıflandırma modellerinde iyi bir metriktir.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(x_test, y_test))

# Grafik Çizimleri
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))
print("\nTest loss: %.1f%%" % (100.0 * loss))
