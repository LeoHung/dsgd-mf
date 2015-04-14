from sys import argv
from pyspark import SparkContext
import numpy as np
from random import random, gauss
import math
import os
from datetime import datetime


def get_n(data):
    return data.map(lambda x: (0, x[1])).reduce(lambda x, y: max(x, y))[1] + 1


def get_m(data):
    return data.map(lambda x: (0, x[2])).reduce(lambda x, y: max(x, y))[1] + 1


IS_TEST = True
test_output_path = "/tmp/netflix/"


def dump_WH(path, w, h, N, M, num_factors, epoch):
    dump_W("%sW_%d.csv" % (path, epoch), w, N, num_factors)
    dump_H("%sH_%d.csv" % (path, epoch), h, M, num_factors)


def dump_W(filename, w, N, num_factors):
    start_time = datetime.now()

    w_vectors = w.map(lambda x: (x[1], x[2])).collect()

    w_matrix = np.zeros((N, num_factors))
    for i, w_vector in w_vectors:
        w_matrix[i] = w_vector

    np.savetxt(filename, w_matrix, fmt='%.4f', delimiter=",")

    print "dump_W : %f secs" % (datetime.now() - start_time).total_seconds()


def dump_H(filename, h, M, num_factors):
    start_time = datetime.now()

    h_vectors = h.map(lambda x: (x[1], x[2])).collect()

    h_matrix = np.zeros((M, num_factors))
    for j, h_vector in h_vectors:
        h_matrix[j] = h_vector

    np.savetxt(filename, h_matrix.T, fmt='%.4f', delimiter=",")

    print "dump_H : %f secs" % (datetime.now() - start_time).total_seconds()


def gen_block_i(i, N, d):
    return i / int(math.ceil(float(N) / float(d)))


def gen_block_j(j, M, d, d_i):
    block_j = j / int(math.ceil(float(M) / float(d)))
    block_j = (block_j + d_i) % d
    return block_j


def new_new_sum_lnzsl(v, w, h, num_workers, N):
    w_dict = w.map(lambda x: (x[1], x[2])).collectAsMap()
    h_dict = h.map(lambda x: (x[1], x[2])).collectAsMap()

    def cal_lnzsl(t):
        i, j, r = t[1][1], t[1][2], t[1][3]
        lnzsl = (r - np.inner(w_dict[i], h_dict[j])) ** 2
        return lnzsl

    return v.map(cal_lnzsl).reduce(lambda x, y: x + y)

def new_sum_lnzsl(v, w, h, num_workers, N):
    def w_key(t):
        i = t[1]
        return gen_block_i(i, N, num_workers)

    def cal_lnzsl(t):
        vs, ws, hs = t[1][0], t[1][1], t[1][2]
        w_dict = {}
        h_dict = {}

        partial_lnzsl = 0.0

        for w in ws:
            w_dict[w[1]] = w[2]

        for h in hs:
            h_dict[h[1]] = h[2]

        for v in vs:
            i, j, r = v[1], v[2], v[3]
            print v
            partial_lnzsl += (r - np.inner(w_dict[i], h_dict[j])) ** 2

        return partial_lnzsl


    total_lnzsl = v.groupWith(
                        w.keyBy(w_key),
                        h.flatMap(lambda t: [(partition_i, t) for partition_i in range(num_workers)])
                    ).map(cal_lnzsl).reduce(lambda x, y: x + y)

    return total_lnzsl


def sum_lnzsl(v, w, h, num_workers, N):
    delta_N = math.ceil(float(N) / float(num_workers))

    def cal_lnzsl(t):
        vs, ws, hs = t[1][0], t[1][1], t[1][2]
        w_dict = {}
        h_dict = {}

        partial_lnzsl = 0.0

        for w in ws:
            w_dict[w[1]] = w[2]

        for h in hs:
            h_dict[h[1]] = h[2]

        for v in vs:
            i, j, r = v[1][1], v[1][2], v[1][3]
            partial_lnzsl += (r - np.inner(w_dict[i], h_dict[j])) ** 2

        return partial_lnzsl

    total_lnzsl = 0.0

    # total_lnzsl += v.keyBy(lambda x: x[1] % 10).groupWith(w.keyBy(lambda x: x[1] % 10), h.flatMap(lambda x: [ (_, x) for _ in range(10)])).map(cal_lnzsl).reduce(lambda x, y: x+ y)

    total_lnzsl += v.keyBy(lambda x: 1).groupWith(w.keyBy(lambda x: 1), h.keyBy(lambda x:1)).map(cal_lnzsl).reduce(lambda x, y : x + y)

    # for i in xrange(num_workers):
    #     sub_v = v.filter(lambda x: delta_N * i <= x[1] and x[1] < delta_N * (i + 1)).keyBy(lambda x: (x[2] / delta_N))
    #     sub_w = w.filter(lambda x: delta_N * i <= x[1] and x[1] < delta_N * (i + 1)).flatMap(lambda x: [(i, x) for i in xrange(num_workers)])
    #     sub_h = h.keyBy(lambda x: (x[1] / delta_N))

    #     total_lnzsl += sub_v.groupWith(sub_w, sub_h).map(cal_lnzsl).reduce(lambda x, y: x + y)

    return total_lnzsl


def rmse(total_lnzsl, total_v):
    from math import sqrt
    return sqrt(total_lnzsl / total_v)


def main(
        num_factors, num_workers, num_iterations, beta_value, lambda_value,
        input_v_filepath, output_w_filepath, output_h_filepath):

    if "Master" in os.environ:
        sc = SparkContext(os.environ['Master'], "DSGD")
    else:
        sc = SparkContext("local[4]", "DSGD")
        # sc = SparkContext("local[%d]" % (num_workers), "DSGD")

    d = num_workers
    # load v
    v = sc.textFile(input_v_filepath)

    def split_data(line):
        tmp = line.split(",")
        return ('v', int(tmp[0]) - 1, int(tmp[1]) - 1, int(tmp[2]))
    v = v.map(lambda line: split_data(line))
    v.cache()

    total_v = v.count()

    # get m, n
    N, M = get_n(v), get_m(v)

    # split V
    def v_key(t):
        i = t[1]
        return gen_block_i(i, N, d)

    # generate w with Ni
    def gen_w(t):
        i, Ni = t[0], t[1]
        return ('w', i, np.array([0.01 * gauss(0, 1) for _ in range(num_factors)]), Ni)
    w = v.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).map(gen_w)

    # generate h with Hj
    def gen_h(t):
        j, Nj = t[0], t[1]
        return ('h', j, np.array([0.01 * gauss(0, 1) for _ in range(num_factors)]), Nj)
    h = v.map(lambda x: (x[2], 1)).reduceByKey(lambda x, y: x + y).map(gen_h)

    # partition v
    v = v.keyBy(v_key).partitionBy(d)
    v.cache()

    all_lnzsl = []
    all_rmse = []

    # main mf loop
    for epoch in xrange(num_iterations):

        print " --------- EPOCH %d --------- " % epoch
        start_time = datetime.now()

        # generate stratum
        for d_i in xrange(d):
            def w_key(t):
                return v_key(t)

            def h_key(t):
                j = t[1]
                return gen_block_j(j, M, d, d_i)

            def sgd_partitions(iterator):
                w_dict = {}
                h_dict = {}
                Ni = {}
                Nj = {}

                n = epoch * total_v + (total_v * d_i) / d  # estimate 'n'
                _beta_value = beta_value
                _lambda_value = lambda_value

                _vs = None
                _ws = None
                _hs = None

                ret = []

                for it in iterator:
                    block_i = it[0]
                    _vs = it[1][0]
                    _ws = it[1][1]
                    _hs = it[1][2]

                    if _vs is None:
                        return []

                    for _w in _ws:
                        w_dict[_w[1]] = _w[2]
                        Ni[_w[1]] = _w[3]

                    for _h in _hs:
                        h_dict[_h[1]] = _h[2]
                        Nj[_h[1]] = _h[3]

                    for _v in _vs:
                        i, j, r = _v[1], _v[2], _v[3]

                        block_j = gen_block_j(j, M, d, d_i)
                        if block_i != block_j:
                            continue

                        epsilon = math.pow((100 + n), - _beta_value)
                        n += 1

                        esti = float(r) - np.inner(w_dict[i], h_dict[j])

                        delta_w = -2.0 * esti * h_dict[j] + 2.0 * ((_lambda_value) / Ni[i]) * w_dict[i]
                        delta_h = -2.0 * esti * w_dict[i] + 2.0 * ((_lambda_value) / Nj[j]) * h_dict[j]

                        w_dict[i] = w_dict[i] - epsilon * delta_w
                        h_dict[j] = h_dict[j] - epsilon * delta_h

                    for i, vector in w_dict.items():
                        ret.append(('w', i, vector, Ni[i]))
                    for j, vector in h_dict.items():
                        ret.append(('h', j, vector, Nj[j]))

                return ret

            new_w_h = v.groupWith(w.keyBy(w_key).partitionBy(d), h.keyBy(h_key).partitionBy(d)).mapPartitions(sgd_partitions)
            # new_w_h = v.cogroup(w.keyBy(w_key).partitionBy(d), h.keyBy(h_key).partitionBy(d), d).mapPartitions(sgd_partitions)

            new_w_h.cache()

            w = new_w_h.filter(lambda x: x[0] == 'w')
            w.cache()
            h = new_w_h.filter(lambda x: x[0] == 'h')
            h.cache()

        if IS_TEST:
            test_start_time = datetime.now()
            total_lnzsl = new_new_sum_lnzsl(v, w, h, num_workers, N)
            all_lnzsl.append(total_lnzsl)
            all_rmse.append(rmse(total_lnzsl, total_v))

            test_duration = datetime.now() - test_start_time
            print "Epoch %d [%f] - (lnzse: %f), (rmse: %f)" % (epoch, test_duration.total_seconds(), all_lnzsl[-1], all_rmse[-1])

        duration = datetime.now() - start_time
        print "Epoch %d [%f]" % (epoch, duration.total_seconds())

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
