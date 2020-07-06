import os
import cv2
import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.optimizers as KO
import threading
import time


###################################
# Data Generator
###################################


class Dataset(object):
    def __init__(self):
        self.images = []
        self.LR_images = []
        self.IHR_images = []

    def load_image(self, path_images):
        for filename in os.listdir(path_images):
            img = cv2.imread(os.path.join(path_images, filename), cv2.IMREAD_GRAYSCALE)
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            if img is not None:
                self.images.append(img)

    def get_image(self, id_image):
        assert id_image < len(self.images)
        return self.images[id_image]

    def prepare(self):
        for image in self.images:
            img = cv2.resize(image, (image.shape[0] // 4, image.shape[1] // 4))
            self.LR_images.append(img)
            img = cv2.resize(img, (img.shape[0] * 4, img.shape[1] * 4))
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            self.IHR_images.append(img)

    def get_Low_resolution_images(self):
        return self.LR_images

    def get_High_resolution_Interpolated_images(self):
        return self.IHR_images

    def get_High_resolution_images(self):
        return self.images


def get_image_batch(train_list, label_list, offset, BATCH_SIZE):
    target_list_train = train_list[offset:offset+BATCH_SIZE]
    target_list_label = label_list[offset:offset+BATCH_SIZE]
    input_list = []
    gt_list = []
    for i in range(0, len(target_list_train)):
        input_img = target_list_train[i]
        gt_img = target_list_label[i]
        input_list.append(np.array(input_img))
        gt_list.append(np.array(gt_img))
    input_list = np.array(input_list)
    input_list.resize([BATCH_SIZE, 300, 300, 1])
    gt_list = np.array(gt_list)
    gt_list.resize([BATCH_SIZE, 300, 300, 1])
    return input_list, gt_list


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def data_generator(train_list, label_list, BATCH_SIZE):
    while True:
        for step in range(len(train_list) // BATCH_SIZE):
            offset = step*BATCH_SIZE
            batch_x, batch_y = get_image_batch(train_list, label_list, offset, BATCH_SIZE)
            yield (batch_x, batch_y)


######################################
# Model Very Deep super resolution
######################################


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def accuracy_function(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def prepare_data(list_images):
    list_images = np.array(list_images)
    return list_images


class VDSR():
    def __init__(self, input_size, n_layers=20, k=3, n_filters=64):
        self.input_size = input_size
        self.n_layers = n_layers
        self.k = k
        self.n_filters = n_filters
        self.BATCH_SIZE = 5
        self.model = self.build()

    def build(self):
        input_image = KL.Input(shape=self.input_size)
        x = KL.Conv2D(self.n_filters, (self.k, self.k), padding='same', kernel_initializer='he_normal')(input_image)
        x = KL.Activation('relu')(x)
        for i in range(0, self.n_layers-2):
            x = KL.Conv2D(self.n_filters, (self.k, self.k), padding='same', kernel_initializer='he_normal')(x)
            x = KL.Activation('relu')(x)
        x = KL.Conv2D(1, (self.k, self.k), padding='same', kernel_initializer='he_normal')(x)
        output_image = KL.add([x, input_image])
        model = KM.Model(input_image, output_image, name='Very deep super resolution')
        return model

    def compile(self):
        # self.model.compile(KO.Adam(lr=0.00001), loss='mse', metrics=[accuracy_function, 'accuracy'])
        # self.model.compile(KO.Adam(lr=0.00001), loss='mse', metrics=['accuracy'])
        self.model.compile(KO.SGD(lr=1e-5, momentum=0.9, decay=1e-4, nesterov=False), loss='mse', metrics=['accuracy'])
        self.model.summary()

    def train(self, data_set):
        self.compile()
        print('Start training ....')
        start_time = time.time()
        train_data = prepare_data(data_set.get_High_resolution_Interpolated_images())
        label_data = prepare_data(data_set.get_High_resolution_images())
        self.model.fit(train_data, label_data, validation_split=0.1, epochs=1000, batch_size=self.BATCH_SIZE, verbose=1)
        print('Done training in ', time.time() - start_time)
        self.model.save('vdsr_trained.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

    def run(self, image):
        image = np.expand_dims(image, axis=0)
        result = self.model.predict(image)
        return result
