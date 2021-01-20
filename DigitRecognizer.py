from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import mnist 

(X_train, y_train), (X_test, y_test) = mnist.load_data();

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32');
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32');
X_train = X_train / 255.0;
X_test = X_test / 255.0;
y_train = to_categorical(y_train,10);
y_test = to_categorical(y_test,10);

model = Sequential();
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)));
model.add(Conv2D(64, (3, 3), activation='selu',kernel_initializer='he_uniform'));
model.add(MaxPooling2D((2, 2)));
model.add(Dropout(0.25));
model.add(Flatten());
model.add(Dense(356, activation='relu',kernel_initializer='he_uniform'));
model.add(Dropout(0.5));
model.add(Dense(10, activation='softmax',kernel_initializer='he_uniform'));

model.compile(optimizer='adam' , 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"]);

history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=5,
    validation_data=(X_test,y_test)
);

score=model.evaluate(X_test,y_test,verbose=0);
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist.h5')

