from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_dir = 'F:\\视听数据\\dataset\\audios_spec_train'
validation_dir = 'F:\\视听数据\\dataset\\audios_spec_test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(257, 200),
    color_mode='grayscale',
    batch_size=4,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(257, 200),
    color_mode='grayscale',
    batch_size=4,
    class_mode='categorical'
)
