from keras import layers
from keras import models

image_model = models.Sequential()
image_model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu',
                              padding='same', input_shape=(224, 224, 3)))
image_model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
image_model.add(layers.MaxPooling2D((2, 2)))
# second
image_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
image_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
image_model.add(layers.MaxPooling2D((2, 2)))
# third
image_model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
image_model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
image_model.add(layers.MaxPooling2D((2, 2)))
# fourth
image_model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
image_model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
# 到这里输出应该是14×14×512
image_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
image_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

# audio input 275*200*1
audio_model = models.Sequential()
audio_model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu',
                              padding='same', input_shape=(257, 200, 1)))
audio_model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
audio_model.add(layers.MaxPooling2D((2, 2)))
# second
audio_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
audio_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
audio_model.add(layers.MaxPooling2D((2, 2)))
# third
audio_model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
audio_model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
audio_model.add(layers.MaxPooling2D((2, 2)))
# fourth
audio_model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
audio_model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
# 到这里输出应该是16×12×512
audio_model.add(layers.MaxPooling2D((16, 12)))
audio_model.add(layers.Dense(128, activation='relu'))
audio_model.add(layers.Dense(128, activation='relu'))

image_out=image_model.output
audio_out=audio_model.output
# AVC merge
avc_model = models.Sequential()
avc_model.add(layers.Dot([image_out, audio_out], axes=4, normalize=False))
avc_model.add(layers.Conv2D(1, (1, 1), activation='relu', padding='same'))
avc_model.add(layers.Dense(1, activation='sigmoid'))
# 在这里就可以判断哪里有乐器块了
avc_model.add(layers.MaxPooling2D(14, 14))
