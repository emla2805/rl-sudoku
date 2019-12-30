import tensorflow as tf
import pandas as pd


def load_dataset(subsample=10000):
    dataset = pd.read_csv("sudoku.csv", sep=',')
    my_sample = dataset.sample(subsample)
    train_set, test_set = create_sudoku_tensors(my_sample)
    return train_set, test_set


def one_hot_encode(quizz):
    ind = tf.constant([int(x) - 1 if int(x) > 0 else int(-1) for x in quizz], dtype=tf.int32)
    ind_one_hot = tf.one_hot(ind, depth=9, dtype=tf.int32)
    return ind_one_hot


if __name__ == '__main__':
    # dataset = pd.read_csv("sudoku.csv", sep=',')
    # print(dataset.quizzes[0])
    # print(dataset.solutions[0])

    quizz = "004300209005009001070060043006002087190007400050083000600000105003508690042910300"
    solution = "864371259325849761971265843436192587198657432257483916689734125713528694542916378"
    print(quizz)
    print(solution)
    quizz_one_hot = one_hot_encode(quizz)
    solution_one_hot = one_hot_encode(solution)
    print(quizz_one_hot)
    print(solution_one_hot)

    index = (2, 3)
    print(quizz_one_hot[index] == 1)

    s = tf.reduce_sum(quizz_one_hot)
    ss = tf.reduce_sum(solution_one_hot)
    print(s)
    print(ss)
