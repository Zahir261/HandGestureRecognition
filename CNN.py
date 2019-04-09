from keras.models import Sequential
from keras.layers import Convolution2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import numpy as np
import PIL
from PIL import Image
from keras import optimizers
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Model
from keras import applications
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model = Sequential()

# Step 1 - Convolution

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 1)))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# Adding a second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))
#sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
# Compiling the CNN
sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
from keras.utils import plot_model
plot_model(model, to_file='model.png')

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)


training_set = train_datagen.flow_from_directory('ReducedNoise/Training_Set',
                                                 color_mode='grayscale',
                                                 target_size=(50, 50),
                                                 batch_size=64,
                                                 class_mode='categorical')

test_set = train_datagen.flow_from_directory('ReducedNoise/Test_Set',
                                            color_mode='grayscale',
                                            target_size=(50, 50),
                                            batch_size=64,
                                            class_mode='categorical')


earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
mcp_save = ModelCheckpoint('mymodel_gray_updated.h5', save_best_only=True, verbose=1, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max')
history = model.fit_generator(training_set,
                         steps_per_epoch=16000 // 64,
                         epochs=100,
                         validation_data=test_set,
                         validation_steps= 4000 // 64,
                         callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

predict_data=ImageDataGenerator(rescale=1. / 255).flow_from_directory('ReducedNoise/Test_Set',
                                            color_mode='grayscale',
                                            target_size=(50, 50),
                                            shuffle=False,
                                            batch_size=64,
                                            subset='validation',
                                            class_mode=None)
preds = model.predict_generator(predict_data, 4000//64)
cm = confusion_matrix(test_set.classes, np.argmax(preds, axis=1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cm_plot_labels = ['1', '2','3','4', '5','6','7', '8','9','10']
plot_confusion_matrix(cm, cm_plot_labels, title='cfmatrixx')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()