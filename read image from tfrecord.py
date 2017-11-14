# encoding=utf-8
import tensorflow as tf
# import re
import os
import linecache
import matplotlib.pyplot as plt;
from PIL import Image
import numpy as np
import scipy.misc

def conv_net(input):
    # input = tf.reshape(features, [-1, 752, 480, 1])
    conv1 = tf.layers.conv2d(input, 64, 7, activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 128, 5, activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, 256, 5, activation=tf.nn.relu)
    conv3_1 = tf.layers.conv2d(conv3, 256, 3, activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3_1, 512, 3, activation=tf.nn.relu)
    conv4_1 = tf.layers.conv2d(conv4, 512, 3, activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu)
    conv5_1 = tf.layers.conv2d(conv5, 512, 3, activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5_1, 1024, 3, activation=tf.nn.relu)
    fc1 = tf.contrib.layers.flatten(conv6)
    logits = tf.layers.dense(fc1, 2)
    return logits

def read_images():
    folder = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/data/'
    imagepaths, labels = list(), list()
    for img in os.listdir(folder):
        print (img)
        imagepaths.append(folder + img)
        labels.append(1)
    return imagepaths, labels

def read_and_decode(file_name):
    filename_queue = tf.train.string_input_producer(file_name)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.float32)
                                       })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [480, 752])
    # image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    print (image)
    print (label)
    # print (np.shape(image))
    return image, label


filename = ["/home/shr/software/softwarebackup/"
            "EUROC/V1_02_medium/mav0/euroc_v102_medium_cam0.tfrecords"]
image, label = read_and_decode(filename)

with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(5):
        img, l = sess.run([image, label])#在会话中取出image和label
        # img = Image.fromarray(example, 'RGB')#这里Image是之前提到的
        print (np.shape(img))

        img = Image.fromarray(np.asarray(img))  # param mode: Mode to use (will be determined from type if None)
        img.save('/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0'
                 +str(i)+'_''Label_'+str(l)+'.jpg')#存下图片

        print(l)
    coord.request_stop()
    coord.join(threads)

# example_batch, label_batch = tf.train.shuffle_batch(
#     [image, label], batch_size=32, capacity=1000+64,
#     min_after_dequeue=1000)

example_batch, label_batch = tf.train.batch(
    [image, label], batch_size=32, capacity=1000+64)

print (example_batch)
print (label_batch)

# if __name__ == '__main__':
    # imagepaths, labels = read_images()
    # print (np.shape(imagepaths))
    # print (np.shape(labels))
    # imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    # labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    #
    # image, label = tf.train.slice_input_producer([imagepaths, labels],
    #                                             shuffle=False)
    # image = tf.read_file(image)
    # image = tf.image.decode_jpeg(image, channels=1)
    # image = tf.image.resize_images(image, [224, 224])
    #
    # X, Y = tf.train.batch([image, label], batch_size=64,
    #                       capacity=64* 8,
    #                       num_threads=4)
    # print (X)
    # print (Y)
    #
    # logits_train = conv_net(X)
    # loss_op = loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    # logits=logits_train, labels=Y))
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # train_op = optimizer.minimize(loss_op)
    #
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    #
    # with tf.Session() as sess:
    #
    #     # Run the initializer
    #     sess.run(init)
    #
    #     # Start the data queue
    #     tf.train.start_queue_runners()
    #
    #     # Training cycle
    #     for step in range(1, 101):
    #         sess.run(train_op)


