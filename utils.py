import tensorflow as tf


def load_dataset(file_path):
    def process(x, y):
        return one_hot_encode(x), one_hot_encode(y)

    dataset = tf.data.experimental.CsvDataset(
        file_path,
        record_defaults=[
            tf.string,
            tf.string,
        ],
        header=True,
    )
    dataset = dataset.map(process)
    ds_eval = dataset.take(30).prefetch(2)

    ds_train = dataset.skip(30)
    ds_train = ds_train.shuffle(1000)
    ds_train = ds_train.prefetch(2)
    return ds_train, ds_eval


def one_hot_encode(quizz):
    chars = tf.strings.to_number(tf.strings.bytes_split(quizz), out_type=tf.int32)
    chars -= 1
    ind_one_hot = tf.one_hot(chars, depth=9, dtype=tf.int32)
    return ind_one_hot

