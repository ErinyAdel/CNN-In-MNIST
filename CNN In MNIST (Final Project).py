import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

"""
 * Load Mnist data that contain digits to classify
 * x --> image , y --> category (numbers in images) 
 """
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
 * Reshape Images Because: Keras / Tensorflow Has Neural Network Accept Data In Specific Shape
 * print(x_train.shape) ---> (60000, 28, 28)
 * (-1) Means: The End Num(no.of_items = -1 OR 60000, height = 28px, width = 28px, color_cannels = 1)
 *                        (batch_size  = -1 OR 10000, height = 28px, width = 28px, color_cannels = 1) 
 *                                                                                 1 Means Grey Not RGB (3)
"""
x_train = x_train.reshape(60000, 28, 28, 1) 
x_test  = x_test.reshape(10000, 28, 28, 1)


""" Normalization / Scaling --> All Mnist Data Are From 0:255 (Make It From 0:1) """
x_train = x_train / 255
x_test = x_test / 255

# handle target categorical data
"""
 * Encode labels/Categories To One-hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0] in matrix)
 * to_categorical: Convert to One-Hot-Encoding (Text To Digits (0's, 1's))
                   = LabelEncoder in sklearn --> Label Encoder   (Vector)
                   = get_dummie   in pandas  --> One-Hot Encoder (Matrix)
"""
# print(y_train.shape) --> (60000,)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape) --> (60000, 10)


""" Build CNN model """
""" 
    * (N + 2P - F) / S + 1
    * Conv --> (28 + 2*0 - 4)/ 1 + 1 = 13    THEN     MaxPooling --> (13 + 2*0 - 2) / 2 + 1 = 6
    * Use MaxPool2D: To Avoid Multiplication Of Matrices (Image Matrix And Filter Matrix)
    * Convolution Network + MaxPooling = Layer (i) -- Untill Dimentions decrese 
    
    * Dropout: Delete From The Neural (NOT USED) (Dropout(0.2)) = Delete 20%
    
    * Conv2D   : specifies the number of filters used in our convolution operation.
    * MaxPool2D: specifies the size of the convolutional filter in pixels.
    * Flatten  : Convert Matrix To Vector (Matrix (28x28) = Vectior (784x1)) To Be Able To Create Neural Network 
    *            specifies how the convolutional filter should step along the x-axis and y-axis of the source image.
    * 4. activation parameter which specifies the name of the activation function you want to apply after performing convolution.
    * Last Dense Must = Num_Classes (10) && softmax Because it's classification.
"""
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2))) # Setting Filter = 2, Stride_Filter = 2 And No Multiplication
model.add(Flatten()) # Customize itself
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

""" 
    Fit The Model To Data 
    * model.fit: Training loop will check at end of every epoch whether the loss is no longer decreasing
    ∟ Once it's found no longer decreasing, model.stop_training is marked True and the training terminates
                       (Stop training when a monitored metric has stopped improving).
    
    Note: Accuracy & Overfitting --> Related With Epochs Number
    
"""
early_stopping = EarlyStopping(monitor="val_loss")
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), callbacks=[early_stopping])

""" Evaluate Model Accuracy """
model.evaluate(x_test, y_test)

""" Test The Model Prediction (Plotting) """
plt.imshow(x_test[9].reshape(28, 28), cmap="binary")
print(np.argmax(model.predict(x_test[9].reshape(1, 28, 28, 1))>0.5).astype("int32"))
