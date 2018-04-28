import tensorflow as tf
import numpy as np

def  read_csv():
    filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue) # 每次都会读取一行内容

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[1], [1], [1], [1], [1]] # 设置每一行缺失值的默认值
    col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
    '''decode_csv 操作会解析这一行内容并将其转为张量列表。'''
    features = tf.concat(0, [col1, col2, col3, col4])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1200):
            # Retrieve a single instance:
            example, label = sess.run([features, col5])

        coord.request_stop()
        coord.join(threads)