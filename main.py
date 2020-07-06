import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from model import *

print('Train or Test ?')
option = input()
if option == 'Train':
    data_set = Dataset()
    data_set.load_image('./Data')
    data_set.prepare()

    input_size = [300, 300, 1]
    vdsr_model = VDSR(input_size)
    vdsr_model.train(data_set)
else:
    print('Path to image: ')
    path = input()
    input_size = [300, 300, 1]
    vdsr_model = VDSR(input_size)
    vdsr_model.load_weights('./vdsr_trained.h5')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Input', img)
    img_down = cv2.resize(img, (img.shape[0]//4, img.shape[1]//4))
    cv2.imshow('Down sample 4', img_down)
    img_inter = cv2.resize(img_down, (img_down.shape[0]*4, img_down.shape[1]*4))
    cv2.imshow('Resize img interpolated', img_inter)
    img_inter = np.reshape(img_inter, (img_inter.shape[0], img_inter.shape[1], 1))
    img_srcnn = vdsr_model.run(img_inter)
    img_srcnn = np.reshape(img_srcnn, (img_srcnn.shape[1], img_srcnn.shape[2], 1))
    img_srcnn = np.array(img_srcnn, dtype=np.uint8)
    cv2.imshow('Very deep super resolution', img_srcnn)
    cv2.waitKey()
    cv2.destroyAllWindows()
