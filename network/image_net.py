from keras import layers
from keras import models

image_model = models.Sequential()

image_model.add(layers.Conv2D(64, (3,3), stride = 2, activation = 'relu', 
                        padding = 'same', input_shape = (224,224,3)))
image_model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
image_model.add(layers.MaxPooling2D((2,2)))
#second
image_model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
image_model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
image_model.add(layers.MaxPooling2D((2,2)))
#third
image_model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
image_model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
image_model.add(layers.MaxPooling2D((2,2)))
#fourth
image_model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
image_model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
#到这里输出应该是14×14×512

image_model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
image_model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))