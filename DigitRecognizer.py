import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
#loading data

# read the data and store data in DataFrame titled train and test
(X_train, y_train), (X_test, y_test) = mnist.load_data();

# Reshape to be samples*pixels*width*height
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32');
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32');

y_train = np_utils.to_categorical(y_train);
y_test = np_utils.to_categorical(y_test);

# convert from integers to floats
X_train = X_train.astype('float32');
X_test = X_test.astype('float32');
# normalize to range [0,1]
X_train = X_train / 255.0;
X_test = X_test / 255.0;

#create a simple model and complie the model
model = Sequential();
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)));
model.add(Conv2D(64, (3, 3), activation='selu'));
model.add(MaxPooling2D((2, 2)));
model.add(Dropout(0.25));
model.add(Flatten());
model.add(Dense(128, activation='selu'));
model.add(Dropout(0.5));
model.add(Dense(10, activation='softmax'));

# Compile the model
model.compile(optimizer='adam' , 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"]);

history = model.fit(
    X_train,
    y_train,
    batch_size=200,
    epochs=10,
    validation_data=(X_test,y_test)
);

score=model.evaluate(X_test,y_test,verbose=0);
print('Test loss:', score[0])
print('Test accuracy:', score[1])

history_df=pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()
