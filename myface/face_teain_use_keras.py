from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.layers import MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import load_model
from load_face_dataset import load_dataset, resize_image


class DataSet:
    def __init__(self, path_name):
        self.train_images = None
        self.train_labels = None
        self.valid_images = None
        self.valid_labels = None
        self.test_images = None
        self.test_labels = None

        self.path_name = path_name

        self.input_shape = None

    def load(self, img_rows=64, img_cols=64, img_channels=3, nb_class=4):
        images, labels = ReadPath(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))

        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)

        self.input_shape = (img_rows, img_cols, img_channels)
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        train_labels = np_utils.to_categorical(train_labels, nb_class)
        valid_labels = np_utils.to_categorical(valid_labels, nb_class)
        test_labels = np_utils.to_categorical(test_labels, nb_class)

        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        train_images /= 255
        valid_images /= 255
        test_images /= 255

        print(train_images)
        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels


class Model:
    def __init__(self):
        self.model = None
        self.batch_size = 32

    def bulid_model(self, dataset, nb_class=4):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=dataset.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_class))
        self.model.add(BatchNormalization())
        self.model.add(Activation('softmax'))
        self.model.summary()
        pass

    def train(self, dataset, nb_epoch=5, data_augmentation=True):
        print('Training ......')
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        if not data_augmentation:
            self.model.fit(dataset.train_images, dataset.train_labels, batch_size=self.batch_size, epochs=nb_epoch, validation_data=(dataset.valid_images, dataset.valid_labels), shuffle=True)
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0）
                samplewise_center=False,  # 是否使输入数据的每个样本的均值为0
                featurewise_std_normalization=False,  # 是否使数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片旋转的角度（0-180）
                width_shift_range=0.2,  # 数据提升时图片水平偏移的角度（单位是图片宽度占比，0-1的浮点数）
                height_shift_range=0.2,  # 数据提升时图片垂直偏移的角度（单位是图片高度占比，0-1的浮点数）
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False  # 是否进行随机垂直翻转
            )
            # 计算整个训练样本的集数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)
            # 利用生成器开始训练模型
            self.model.fit_generator(
                datagen.flow(dataset.train_images, dataset.train_labels, batch_size=self.batch_size),
                steps_per_epoch=dataset.train_images.shape[0],
                epochs=nb_epoch,
                validation_data=(dataset.valid_images, dataset.valid_labels)
            )

        pass

    MODEL_PATH = 'me.face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
        pass

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
        pass

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        pass

    def face_predict(self, image):
        IMAGE_SIZE = 64
        image = resize_image(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image, self.batch_size)
        print('#####result is ', result)
        aa = self.model.predict(image, self.batch_size)
        print('tse--result is ', aa)
        result = (result > 0.75).astype('int32')
        print('22222result is ', result)
        print('**************')
        for popleNum in range(len(result[0])):
            if 1 == result[0][popleNum]:
                break
        ret = popleNum
        if 0 == result[0][popleNum]:
            ret = 0xff

        return ret


if __name__ == '__main__':
    dataset = DataSet('./mypic')
    dataset.load()

    model = Model()
    model.bulid_model(dataset)

    model.train(dataset)
    model.save_model()

    model = Model()
    model.load_model()
    model.evaluate(dataset)
