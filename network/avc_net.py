from keras import layers
from keras import models

#这里需要将两个子网络连接起来
avc_model = models.Sequential()

avc_model.add(layers.Conv2D(1, (1,1), activation = 'relu', padding = 'same', input_shape = ( )))
avc_model.add(layers.Dense(1, activation = 'sigmoid'))
#在这里就可以判断哪里有乐器块了

avc_model.add(layers.MaxPooling2D(14,14))
#在这里输出一个概率

from keras import optimizers

avc_model.compile(loss = 'binary_crossentropy',
                    optimizer = optimizers.RMSprop(lr = 1e-4),
                    metrics = ['acc'])

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