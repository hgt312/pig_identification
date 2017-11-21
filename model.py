from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from utils import get_nb_files


# cnn部分，使用了inception resnet v2
# pooling参数: 'max'/'avg'/None
base_model = InceptionResNetV2(
    include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling='max')

# 全连接层以及分类输出
x = base_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(30, activation='softmax')(x)

# 定义模型
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

# 编译模型及选择优化方法
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 准备数据
train_dir = '/media/hgt/share2/jdd/train'  # 训练集数据
dev_dir = '/media/hgt/share2/jdd/dev'  # 验证集数据

nb_classes = 30  # 分类数
nb_train_samples = get_nb_files(train_dir)      # 训练样本个数
nb_val_samples = get_nb_files(dev_dir)       # 验证集样本个数

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

dev_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),  # all images will be resized to 150x150
    batch_size=32,
    class_mode='categorical')

dev_generator = dev_datagen.flow_from_directory(
    dev_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical')


# 训练模型
model.fit_generator(
        train_generator,
        steps_per_epoch=645,
        epochs=1,
        validation_data=dev_generator,
        validation_steps=9000)
model.save_weights('/media/hgt/share2/jdd/try_time_1st.h5')
