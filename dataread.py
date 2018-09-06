import random, os, cv2
import numpy as np

TRAIN_ROOT = '/home/ilab/ten-hospital/data/new/train'
TEST_ROOT = '/home/ilab/ten-hospital/data/new/test'
ROWS = 64 * 4
COLS = 64 * 4
CHANNELS = 3

def read(keepdims = True):
    TRAIN_DIR = []
    for dir in os.listdir(TRAIN_ROOT):
        if dir.startswith('.'):
            continue
        TRAIN_DIR.append(os.path.join(TRAIN_ROOT, dir))
    TEST_DIR = []
    for dir in os.listdir(TEST_ROOT):
        if dir.startswith('.'):
            continue
        TEST_DIR.append(os.path.join(TEST_ROOT, dir))

    train_image = []
    for im_dir in TRAIN_DIR:
        train_image += [os.path.join(im_dir, filename) for filename in os.listdir(im_dir) if not filename.startswith('.')]

    test_image = []
    for im_dir in TEST_DIR:
        test_image += [os.path.join(im_dir, filename) for filename in os.listdir(im_dir) if not filename.startswith('.')]

    # slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
    random.shuffle(train_image)

    train = prep_data(train_image)
    validation = prep_data(test_image)

#    prct = 0.96
#    train_num = int(len(all_image) * prct)
#    train_img = all_image[0 : train_num]
#    validation_img = all_image[train_num:]
#    train = prep_data(train_img)
#    validation = prep_data(validation_img)
    train_labels = get_labels(train_image, keepdims)
    validation_labels = get_labels(test_image, keepdims)

    return (train,validation,train_labels,validation_labels)

# 按固定行列读取图片
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i % 1000 == 0: print('Processed {} of {}'.format(i, count))

    return data

# 根据文件名生成label
def get_labels(image_path, keepdims = True):
    labels = []
    for path in image_path:
        filename = os.path.split(path)[1]
        if filename[0] == '1':
            if keepdims:
                labels.append([[[1, 0]]])
            else:
                labels.append([1, 0])
        elif filename[0] == '2':
            if keepdims:
                labels.append([[[0, 1]]])
            else:
                labels.append([0, 1])
        else:
            print('label set failed')
            exit()

    return np.array(labels)

def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test
