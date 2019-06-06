# Importing the Keras libraries and packages
# CNN for binary classification
#cnnClassification.py
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation
import os,numpy as np

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# # CONV => RELU => POOL
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'sigmoid'))
classifier.add(Dense(units = 4, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#classifier.summary()

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../../Detection/quary/mainQuary/dataset/resized/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('../../Detection/quary/mainQuary/test/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

classifier.fit_generator(training_set,
steps_per_epoch = 311,
epochs = 25,
validation_data = test_set,
validation_steps = 134)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../../Detection/dataset/05-56-31.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print training_set.class_indices
print result


print max(result[0])
indices = np.argmax(result[0])
print indices
