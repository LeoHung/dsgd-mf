from sys import argv
from pyspark import SparkContext
import numpy as np
from random import gauss
import math
import os
from datetime import datetime

# Test mode will calculate Lnzsl and rmse for each iteration.
# It also calculate the running time for profiling
IS_TEST = False


def get_n(data):
    """
        get the largest index of i axis
        @data: RDD of data (format: (marker, i, j))
    """
    return data.map(lambda x: (0, x[1])).reduce(lambda x, y: max(x, y))[1] + 1


def get_m(data):
    """
        get the largest index of j axis
        @data: RDD of data (format: (marker, i, j))
    """
    return data.map(lambda x: (0, x[2])).reduce(lambda x, y: max(x, y))[1] + 1


def dump_W(filename, w, N, num_factors):
    """
        dump the W matrix into file, in dense format
        @filename: output filename
        @w: w dictionary
        @N: the largest index in i axis
        @num_factors: number of factors
    """
    if IS_TEST:
        start_time = datetime.now()

    w_matrix = np.zeros((N, num_factors))
    for i, w_vector in w.items():
        w_matrix[i] = w_vector

    np.savetxt(filename, w_matrix, fmt='%.4f', delimiter=",")

    if IS_TEST:
        print "dump_W : %f secs" % (datetime.now() - start_time).total_seconds()


def dump_H(filename, h, M, num_factors):
    """
        dump the H matrix into file, in dense format
        @filename: output filename
        @h: h dictionary
        @M: the largest index in j axis
        @num_factors: number of factors
    """
    if IS_TEST:
        start_time = datetime.now()

    h_matrix = np.zeros((M, num_factors))
    for j, h_vector in h.items():
        h_matrix[j] = h_vector

    np.savetxt(filename, h_matrix.T, fmt='%.4f', delimiter=",")

    if IS_TEST:
        print "dump_H : %f secs" % (datetime.now() - start_time).total_seconds()


def gen_block_i(i, N, d):
    """
        generate block_i for an entry (i, j)
        @i: i of the entry
        @N: the largest index in i axis
        @d: the number of workers
    """
    return i / int(math.ceil(float(N) / float(d)))


def gen_block_j(j, M, d, d_i):
    """
        generate block_j for an entry (i, j)
        @j: j of the entry
        @M: the largest index in j axis
        @d: the number of workers
        @d_i: current index of sub-iteration
    """
    block_j = j / int(math.ceil(float(M) / float(d)))
    block_j = (block_j + d_i) % d
    return block_j


def cal_lnzsl(v, w, h, num_workers, N):
    """
        calculate the sum of Lnzsl
        @v: RDD of v
        @w: w dictionary
        @h: h dictionary
        @num_workers: number of workers
        @N: the largest index in i axis
    """
    def cal_lnzsl(t):
        i, j, r = t[1][1], t[1][2], t[1][3]
        lnzsl = (r - np.inner(w[i], h[j])) ** 2
        return lnzsl

    return v.map(cal_lnzsl).reduce(lambda x, y: x + y)


def rmse(total_lnzsl, total_v):
    """
        calculate rmse
        @total_lnzsl: sum of Lnzsl
        @total_v: the number of entries
    """
    return math.sqrt(total_lnzsl / total_v)


def load_v(sc, input_v_filepath):
    v = sc.textFile(input_v_filepath)

    def split_data(line):
        tmp = line.split(",")
        return ('v', int(tmp[0]) - 1, int(tmp[1]) - 1, int(tmp[2]))
    v = v.map(lambda line: split_data(line))
    return v


def main(
        num_factors, num_workers, num_iterations, beta_value, lambda_value,
        input_v_filepath, output_w_filepath, output_h_filepath):

    if "Master" in os.environ:
        sc = SparkContext(os.environ['Master'], "DSGD")
    else:
        sc = SparkContext("local[4]", "DSGD")

    d = num_workers

    # load v
    v = load_v(sc, input_v_filepath)
    v.cache()

    total_v = v.count()

    # get m, n
    N, M = get_n(v), get_m(v)

    # generate w with Ni
    Ni = v.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    w = dict([(i, np.array([0.01 * gauss(0, 1) for _ in range(num_factors)])) for i in Ni.keys()])

    # generate h with Hj
    Nj = v.map(lambda x: (x[2], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    h = dict([(j, np.array([0.01 * gauss(0, 1) for _ in range(num_factors)])) for j in Nj.keys()])

    # partition v
    def v_key(t):
        i = t[1]
        return gen_block_i(i, N, d)

    v = v.keyBy(v_key).partitionBy(d)
    v.cache()

    if IS_TEST:
        all_lnzsl = []
        all_rmse = []

    # main mf loop
    for epoch in xrange(num_iterations):
        if IS_TEST:
            print " --------- EPOCH %d --------- " % epoch
            start_time = datetime.now()

        for d_i in xrange(d):
            def sgd_partitions(iterator):
                n = epoch * total_v + (total_v * d_i) / d  # estimate 'n'
                _beta_value = beta_value  # fetch beta_value
                _lambda_value = lambda_value  # fetch lambda_value

                new_w = dict()  # the cached w
                new_h = dict()  # the cached h

                ret = []  # the return list

                for v_point in iterator:
                    i, j, r = v_point[1][1], v_point[1][2], v_point[1][3]

                    # generate block_i, block_j for each entry
                    # only the entries whose block_i == block_j will be
                    # processed (in stratum)
                    block_i = v_point[0]
                    block_j = gen_block_j(j, M, d, d_i)
                    if block_i != block_j:
                        continue

                    # if w[i] is not in cached w, fetch it
                    if i not in new_w:
                        new_w[i] = w[i]
                    # if h[i] is not in cached h, fetch it
                    if j not in new_h:
                        new_h[j] = h[j]

                    epsilon = math.pow((100 + n), - _beta_value)
                    n += 1

                    esti = float(r) - np.inner(new_w[i], new_h[j])

                    delta_w = -2.0 * esti * new_h[j] + 2.0 * ((_lambda_value) / Ni[i]) * new_w[i]
                    delta_h = -2.0 * esti * new_w[i] + 2.0 * ((_lambda_value) / Nj[j]) * new_h[j]

                    new_w[i] = new_w[i] - epsilon * delta_w
                    new_h[j] = new_h[j] - epsilon * delta_h

                # add updated w into return list
                for i, vector in new_w.items():
                    ret.append(("w", i, vector))
                for j, vector in new_h.items():
                    ret.append(("h", j, vector))

                return ret

            new_w_h = v.mapPartitions(sgd_partitions)
            new_w_h.cache()

            new_w = new_w_h.filter(lambda x: x[0] == 'w').map(lambda x: (x[1], x[2])).collectAsMap()
            new_h = new_w_h.filter(lambda x: x[0] == 'h').map(lambda x: (x[1], x[2])).collectAsMap()

            # update w
            for i, w_vector in new_w.items():
                w[i] = w_vector

            # update h
            for j, h_vector in new_h.items():
                h[j] = h_vector

        if IS_TEST:
            # if in test mode: calculate Lnzsl and rmse
            test_start_time = datetime.now()
            total_lnzsl = cal_lnzsl(v, w, h, num_workers, N)
            all_lnzsl.append(total_lnzsl)
            all_rmse.append(rmse(total_lnzsl, total_v))

            test_duration = datetime.now() - test_start_time
            print "Epoch %d [%f] - (lnzse: %f), (rmse: %f)" % (epoch, test_duration.total_seconds(), all_lnzsl[-1], all_rmse[-1])

            duration = datetime.now() - start_time
            print "Epoch %d [%f]" % (epoch, duration.total_seconds())

    # output W and H matrix
    dump_W(output_w_filepath, w, N, num_factors)
    dump_H(output_h_filepath, h, M, num_factors)

    if IS_TEST:
        print "i, total_lnzsl"
        for i, total_lnzsl in enumerate(all_lnzsl):
            print "%d,%f" % (i, total_lnzsl)

        print "--------------"
        print "i, rmse"
        for i, rmse_value in enumerate(all_rmse):
            print "%d,%f" % (i, rmse_value)


if __name__ == "__main__":
    num_factors = int(argv[1])
    num_workers = int(argv[2])
    num_iterations = int(argv[3])
    beta_value = float(argv[4])
    lambda_value = float(argv[5])
    input_v_filepath = argv[6]
    output_w_filepath = argv[7]
    output_h_filepath = argv[8]

    main(
        num_factors=num_factors,
        num_workers=num_workers,
        num_iterations=num_iterations,
        beta_value=beta_value,
        lambda_value=lambda_value,
        input_v_filepath=input_v_filepath,
        output_w_filepath=output_w_filepath,
        output_h_filepath=output_h_filepath)
