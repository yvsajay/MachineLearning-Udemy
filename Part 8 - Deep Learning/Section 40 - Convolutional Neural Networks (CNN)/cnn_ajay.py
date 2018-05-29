#building cnn

# Importing the libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize cnn
classifier = Sequential()

#step 1 convolution
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation ='relu')) #tensorflow backend have reverse order than what is shown in help


#step 2 pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#second convolution layer- levelII
classifier.add(Conv2D(32,(3,3), activation ='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3 flatten
classifier.add(Flatten())
 
#step 4 full connection
classifier.add(Dense(128, activation = 'relu')) #output 128 is from experience
classifier.add(Dense(1,activation = 'sigmoid')) #if output more than 2, than softmax function needs to be used


# step 5 compiler
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #adam = stochastic gradient descent
 
#fitting cnn to images
#use keras.io for image augmentation
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
#steps per epoch is number of images in training set, #validation_steps is number of images in test set

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000)