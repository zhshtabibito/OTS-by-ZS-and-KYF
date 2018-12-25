from keras import layers
from keras import models

audio_model = models.Sequential()
#第一个层组，写得不一定对。网络形状没有变化，所以padding需要置为same
audio_model.add(layers.Conv2D(64, (3,3), stride = 2, activation = 'relu', 
                        padding = 'same', input_shape = (257,200,1)))
audio_model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
audio_model.add(layers.MaxPooling2D((2,2)))
#second
audio_model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
audio_model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
audio_model.add(layers.MaxPooling2D((2,2)))
#third
audio_model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
audio_model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
audio_model.add(layers.MaxPooling2D((2,2)))
#fourth
audio_model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
audio_model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
#到这里输出应该是16×12×512

audio_model.add(layers.MaxPooling2D(16,12))
audio_model.add(layers.Dense(128,activation = 'relu'))
audio_model.add(layers.Dense(128,activation = 'relu'))
fdsf