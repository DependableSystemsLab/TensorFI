###
### Implementation from https://github.com/exelban/tensorflow-cifar-10
### AlexNet on Cifar-10 dataset, dataset from https://www.cs.toronto.edu/~kriz/cifar.html
### 

import numpy as np
import tensorflow as tf
from time import time
import math
import pickle
import numpy as np
import os
from six.moves.urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import TensorFI as ti

global isTrain 
global isTest


# Train or test the model, you first need to train the model
# The model will automatically download the dataset before training.
isTest = False
isTrain = True
 

def get_data_set(name="train"):
    x = None
    y = None

    maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f)
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    return x, dense_to_one_hot(y)

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)

def model():
    filterSize = 8
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=x_image,
            filters=32/filterSize,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, padding='SAME') 
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64/filterSize,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, padding='SAME') 

    with tf.variable_scope('conv2') as scope:
        conv1 = tf.layers.conv2d(
            inputs=pool2,
            filters=128/filterSize,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        ) 
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=128/filterSize,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu
        )

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=128/filterSize,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu
        )


        pool = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, padding='SAME') 

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(pool, [-1, 4 * 4 * 128/filterSize])

        fc1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu) 
        fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu) 
        fc3 = tf.layers.dense(inputs=fc2, units=_NUM_CLASSES, activation=tf.nn.relu) 

        softmax = tf.nn.softmax(fc3) 

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate

def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate

def train(epoch):
    global epoch_start
    epoch_start = time()
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0

    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
 

        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))

    test_and_save(i_global, epoch)

def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")


# Prepreation for training, e.g., setting optimizer
if(isTrain):
    tf.set_random_seed(21)

    train_x, train_y = get_data_set("train")
    test_x, test_y = get_data_set("test")
    x, y, output, y_pred_cls, global_step, learning_rate = model()


    global_accuracy = 0
    epoch_start = 0


    # PARAMS
    _BATCH_SIZE = 128
    _EPOCH = 30
    _CLASS_SIZE = 10
    _SAVE_PATH = "./modelSaver/"
    if not os.path.exists:
        os.makedirs(_SAVE_PATH)
     
    # LOSS AND OPTIMIZER
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08).minimize(loss, global_step=global_step)

    # PREDICTION AND ACCURACY CALCULATION
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # SAVER
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)
# Prepreation for testing, e.g., restoring the trained model
elif(isTest):
    test_x, test_y = get_data_set("test")
    x, y, output, y_pred_cls, global_step, learning_rate = model()


    _BATCH_SIZE = 128
    _CLASS_SIZE = 10
    _SAVE_PATH = "./modelSaver/"


    saver = tf.train.Saver()
    sess = tf.Session()    

    try:
        print("\nTrying to restore last checkpoint ...")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
        saver.restore(sess, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except ValueError:
        print("\nFailed to restore checkpoint. Initializing variables instead.")
        sess.run(tf.global_variables_initializer())


def main():
    global isTrain 
    global isTest

    # you need to first train the model
    if(isTrain):
        train_start = time()

        for i in range(_EPOCH):
            print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
            train(i)

        hours, rem = divmod(time() - train_start, 3600)
        minutes, seconds = divmod(rem, 60)
        mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
        print(mes.format(global_accuracy, int(hours), int(minutes), seconds))

    # after the model is trained, you can perform fault injection
    if(isTest):
        
        # we use the inputs that can be correctly identified by the model for FI
        tx = test_x[:50 ,:]
        ty = test_y[:50, :]
        preds = sess.run(y_pred_cls, feed_dict={x: tx, y: ty})
        correct = (np.argmax(ty, axis=1) == preds)
        correctIndex = np.argwhere(correct == True)
        correctIndex = correctIndex.flatten()
        # correctIndex stores the index of inputs that can be correctly identified 

        X = tx
        Y = ty 
        fi = ti.TensorFI(sess, name = "Perceptron", logLevel = 50, disableInjections = False)

        correct = []
        # save each random FI result into file
        resFile = open("alex-ranFI.csv", "a")

        totalFI = 1000  # number of random FI trials
        for i in range(10):

            # construct single input
            tx = X[ correctIndex[i] , : ]
            ty = Y[ correctIndex[i] , :]
            tx = tx.reshape(1, 3072)
            ty = ty.reshape(1, 10)

            for j in range(totalFI):
                preds = sess.run(y_pred_cls, feed_dict={x: tx, y: ty})

                acy =  (np.argmax(ty, axis=1) == preds)[0] 
                # FI does not result in SDC
                if(acy == True):
                    resFile.write(`1` + ",")
                else:
                    resFile.write(`0` + ",") 

            resFile.write("\n")
            print("data ", i, j, "run")


        #Full batch eval
        '''
        i = 0
        predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
        while i < len(test_x):
            j = min(i + _BATCH_SIZE, len(test_x))
            batch_xs = test_x[i:j, :]
            batch_ys = test_y[i:j, :]
            predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
            i = j
             
        correct = (np.argmax(test_y, axis=1) == predicted_class)
        acc = correct.mean() * 100
        correct_numbers = correct.sum()
        print()
        print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x))) 
        ''' 

if __name__ == "__main__":
    main()


sess.close()
