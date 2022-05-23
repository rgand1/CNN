from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import numpy as np
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

# load train and test dataset
def load_dataset():
	#np.random.seed(0)

	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	#validx = np.random.choice(range(trainX.shape[0]),5000,replace=False)
	#valX = trainX[validx]
	#valY = trainY[validx]
	#trainX = np.delete(trainX,validx,0)
	#trainY = np.delete(trainY,validx,0)

	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	#valY = to_categorical(valY)

	return trainX, trainY, testX, testY#,valX,valY

# scale pixels
def prep_pixels(train, test,customize_mean=False):
	if customize_mean:
		mean = np.mean(train, axis=(1, 2, 3), keepdims=True)
		std = np.std(train, axis=(1, 2, 3), keepdims=True)
		train_norm = (train - mean) / std

		mean = np.mean(test, axis=(1, 2, 3), keepdims=True)
		std = np.std(test, axis=(1, 2, 3), keepdims=True)
		test_norm = (test - mean) / std
		return train_norm, test_norm
	else:
		# convert from integers to floats
		train_norm = train.astype('float32')
		test_norm = test.astype('float32')
		# normalize to range 0-1
		train_norm = train_norm / 255.0
		test_norm = test_norm / 255.0
		# return normalized images
		return train_norm, test_norm


# define cnn model
def define_model(model):
	#Baseline: 1 VGG Block
	if model == 'BaseLine1':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 input_shape=(32, 32, 3)))
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	#Baseline: 2 VGG BLOCK
	if  model == 'BaseLine2':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 input_shape=(32, 32, 3)))
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	#Baseline: 3 VGG BLOCK
	if  model == 'BaseLine3':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
							 input_shape=(32, 32, 3)))
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	if model == 'BaseLine_Dropout' or model == 'BaseLine_Augmentation':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
							 input_shape=(32, 32, 3)))
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dropout(0.2))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	if model == 'BaseLine_L2':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 kernel_regularizer=l2(0.001), input_shape=(32, 32, 3)))
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 kernel_regularizer=l2(0.001)))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 kernel_regularizer=l2(0.001)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 kernel_regularizer=l2(0.001)))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 kernel_regularizer=l2(0.001)))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 kernel_regularizer=l2(0.001)))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001)))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	if model == 'DropoutAugemntantationBN':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 input_shape=(32, 32, 3)))
		model.add(BatchNormalization())
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.3))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.4))
		model.add(Flatten())
		model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model

	if model  == 'Adam':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 input_shape=(32, 32, 3)))
		model.add(BatchNormalization())
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.3))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.4))
		model.add(Flatten())
		model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = Adam
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	if model == 'ResNet':
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
						 input_shape=(32, 32, 3)))
		model.add(BatchNormalization())
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.2))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.3))
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.4))
		model.add(GlobalAveragePooling2D())
		model.add(Flatten())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = Adam
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model






# plot diagnostic learning curves
def summarize_diagnostics(history,model,epOchs):
	epochs = list(range(0,epOchs))

	train_loss = history.history['loss']
	train_acc = history.history['accuracy']

	val_loss = history.history['val_loss']
	val_acc = history.history['val_accuracy']


	fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
	ax[0].plot(epochs, train_loss, color='green', label="Training loss")
	ax[0].plot(epochs, val_loss, color='red', label="Validation loss")

	ax[0].legend()
	ax[0].set(ylabel='Cross Entropy Loss')
	ax[0].grid()

	ax[1].plot(epochs, train_acc, color='green', label="Training accuracy")
	ax[1].plot(epochs, val_acc, color='red', label="Validation accuracy")

	ax[1].legend()
	ax[1].set(xlabel='Epochs', ylabel='Classification Accuracy (%)')
	ax[1].grid()

	# save plot to file
	plt.savefig(model+'_plot.png')
	plt.close()

# run the test harness for evaluating a model
def run_test_harness(models,customize_mean=False):
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX,customize_mean=customize_mean)
	# define model
	epochs = 100
	for mOdel in models:
		model = define_model(mOdel)
		if mOdel == 'BaseLine_Augmentation' or model == 'DropoutAugemntantationBN':
			if mOdel == 'DropoutAugemntantationBN':
				epochs = 400
			# fit model
			datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
			it_train =datagen.flow(trainX,trainY,batch_size=64)
			steps = int(trainX.shape[0]/64)
			history = model.fit_generator(it_train,steps_per_epoch=steps,epochs=epochs,validation_data=(testX,testY),verbose=1)
		else:
			# fit model
			history = model.fit(trainX, trainY, epochs=epochs, batch_size=64, validation_data=(testX, testY), verbose=1)

		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('The accuracy of the model ('+mOdel+')in the test set is> %.3f' % (acc * 100.0))
		# learning curves
		summarize_diagnostics(history,mOdel,epochs)


def main():
	mOdels = ['BaseLine_Augmentation','DropoutAugemntantationBN','Adam']

	run_test_harness(mOdels)

#'BaseLine1','BaseLine2','BaseLine3','BaseLine_Dropout','BaseLine_L2'
if __name__ == "__main__":
    main()
