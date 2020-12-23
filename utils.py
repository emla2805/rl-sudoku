import tensorflow as tf
import numpy as np


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


def create_constraint_mask():
    constraint_mask = np.zeros((81, 3, 81), dtype=int)
    # row constraints
    for a in range(81):
        r = 9 * (a // 9)
        for b in range(9):
            constraint_mask[a, 0, r + b] = 1

    # column constraints
    for a in range(81):
        c = a % 9
        for b in range(9):
            constraint_mask[a, 1, c + 9 * b] = 1

    # box constraints
    for a in range(81):
        r = a // 9
        c = a % 9
        br = 3 * 9 * (r // 3)
        bc = 3 * (c // 3)
        for b in range(9):
            r = b % 3
            c = 9 * (b // 3)
            constraint_mask[a, 2, br + bc + r + c] = 1

    constraint_mask = np.reshape(constraint_mask, [1, 81, 3, 81, 1])
    return constraint_mask


if __name__ == '__main__':
    c = create_constraint_mask()
    q = one_hot_encode(
        "004300209005009001070060043006002087190007400050083000600000105003508690042910300").numpy()
    s = one_hot_encode(
        "864371259325849761971265843436192587198657432257483916689734125713528694542916378").numpy()
    print(q)

    print(c)
    res = s * np.reshape(c, [1, 81, 3, 81, 1])
    res = np.sum(res, axis=3)
    print(res)
    print(res.shape)
    print(np.max(res))
