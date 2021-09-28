from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

X_train, X_test, y_train, y_test = np.load('./numpy_data/binary_image_data.npy')
print(X_train.shape)
print(X_train.shape[0])
print(np.bincount(y_train))



X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

#gpu 메모리 할당
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.22)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

with tf.device('GPU:0'):
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    # 최적화 방식(Adam, SGD, RMSprop)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    #opt = tf.keras.optimizers.RMSprop(learning_rate=0.001) 
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/kongju_others_classify.model"
        
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=0, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)

model.summary()

history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.15, callbacks=[checkpoint, early_stopping] ,verbose=2)

#print("정확도 : %.2f " %(model.evaluate(X_test, y_test)[1]))
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
plt.show()


#########################################################

