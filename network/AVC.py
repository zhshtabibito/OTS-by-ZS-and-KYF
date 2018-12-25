from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator

def main():
    # vision input 224*224*3
    image_model = models.Sequential()
    image_model.add(layers.Conv2D(64, (3, 3), stride=2, activation='relu',
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
    audio_model.add(layers.Conv2D(64, (3, 3), stride=2, activation='relu',
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
    audio_model.add(layers.MaxPooling2D(16, 12))
    audio_model.add(layers.Dense(128, activation='relu'))
    audio_model.add(layers.Dense(128, activation='relu'))

    # AVC merge
    avc_model = models.Sequential()
    avc_model.add(layers.dot([image_model, audio_model], axes=2, normalize=False))
    avc_model.add(layers.Conv2D(1, (1, 1), activation='relu', padding='same', input_shape=( )))
    avc_model.add(layers.Dense(1, activation='sigmoid'))
    # 在这里就可以判断哪里有乐器块了
    avc_model.add(layers.MaxPooling2D(14, 14))
    # 在这里输出一个概率

    from keras import optimizers
    avc_model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.RMSprop(lr = 1e-4),
                        metrics=['acc'])

    #读取数据



    #训练
    history = avc_model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 50
    )
    avc_model.save('OTS_model.h5')
