from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# a database which conteains 60.000 images for training and 10.00 images for test
from keras.datasets import mnist 

# read the data and store data in DataFrame titled train and test
(X_train, y_train), (X_test, y_test) = mnist.load_data();

# The data size is (60.000,28,28), and another dimension is needed to reshape the array to form (60,000,height = 28px, width = 28px , canal = 1)
# Keras requires an extra dimension in the end which correspond to channels. 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32');
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32');

# Normalize the data to reduce the effect of illumination's differences.
# normalize to range [0,1]
X_train = X_train / 255.0;
X_test = X_test / 255.0;

# to_categorical function that converts a vector of integers into a matrix of binary vectors
y_train = to_categorical(y_train);
y_test = to_categorical(y_test);


#create a simple model and complie the model
model = Sequential();

# the first and second layers are convolutional (Conv2D) 
# first argument represents the number of filters: the first layer has 32 filters and the second 64
# kernel_size represents the kernel size
# activation is the activation function used to ignore unimportant features.
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)));
model.add(Conv2D(64, (3, 3), activation='selu'));

# MaxPooling2D - it acts as a sampling filter, looking at the 2 neighboring pixels and choosing the maximum value. 
# They are used to reduce computing costs and overfitting. 
# The received parameter represents the size of the pool.
# At this point the network is able to combine local features and learn more global features of the image
model.add(MaxPooling2D((2, 2)));


# With this layer, 25% of the network neurons are randomly ignored,
# setting their value to 0 for each training sample, thus reducing overfitting.
model.add(Dropout(0.25));

# converts the map of final features into a vector
# it combines all the local characteristics found by the previous convolutional layers.
model.add(Flatten());

model.add(Dense(128, activation='relu'));
model.add(Dropout(0.5));
model.add(Dense(10, activation='softmax'));

# configuration of the learning process
# optimizer='adam' - used to improve parameters minimizing losses
# loss function - measures how poor the model's performance is on known images, ie the error rate between labels and predictions 
# metrics - used to evaluate the performance of the model
model.compile(optimizer='adam' , 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"]);

# model training and iteration through training data in 10 epochs
history = model.fit(
    X_train,
    y_train,
    batch_size=200,
    epochs=10,
    validation_data=(X_test,y_test)
);

# model evaluation - test data were not used in the model drive process so they are new, which means that an accuracy approaching 99% should be obtained
score=model.evaluate(X_test,y_test,verbose=0);
print('Test loss:', score[0])
print('Test accuracy:', score[1])

