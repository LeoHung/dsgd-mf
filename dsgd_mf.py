from sys import argv
from pyspark import SparkContext
import numpy as np
from random import random, gauss
import math


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
    w_vectors = w.collect()
    w_vectors = [(i, vector) for _, i, vector, Ni in w_vectors]

    w = np.zeros((N, num_factors))
    for i, w_vector in w_vectors:
        w[i] = w_vector

    np.savetxt(filename, w, delimiter=",")


def dump_H(filename, h, M, num_factors):
    h_vectors = h.collect()
    h_vectors = [(j, vector) for _, j, vector, Nj in h_vectors]

    h = np.zeros((M, num_factors))
    for j, h_vector in h_vectors:
        h[j] = h_vector

    np.savetxt(filename, h.T, delimiter=",")


def gen_block_i(i, N, d):
    return i / int(math.ceil(float(N) / float(d)))


def gen_block_j(j, M, d, d_i):
    block_j = j / int(math.ceil(float(M) / float(d)))
    block_j = (block_j + d_i) % d
    return block_j

def main(
        num_factors, num_workers, num_iterations, beta_value, lambda_value,
        input_v_filepath, output_w_filepath, output_h_filepath):
    sc = SparkContext("local[10]", "DSGD")

    d = num_workers
    # load v
    v = sc.textFile(input_v_filepath)

    def split_data(line):
        tmp = line.split(",")
        return ('v', int(tmp[0]) - 1, int(tmp[1]) - 1, int(tmp[2]))
    v = v.map(lambda line: split_data(line))

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
        return ('w', j, np.array([0.01 * gauss(0, 1) for _ in range(num_factors)]), Nj)
    h = v.map(lambda x: (x[2], 1)).reduceByKey(lambda x, y: x + y).map(gen_h)

    # vi_min = v.map(lambda x: x[1]).distinct().reduce(lambda x, y: min(x,y))
    # vi_max = v.map(lambda x: x[1]).distinct().reduce(lambda x, y: max(x,y))
    # vj_min = v.map(lambda x: x[2]).distinct().reduce(lambda x, y: min(x,y))
    # vj_max = v.map(lambda x: x[2]).distinct().reduce(lambda x, y: max(x,y))
    # vi_avg = v.map(lambda x: x[1]).distinct().mean()
    # vj_avg = v.map(lambda x: x[2]).distinct().mean()

    # wi_min = w.map(lambda x: x[1]).reduce(lambda x, y: min(x,y))
    # wi_max = w.map(lambda x: x[1]).reduce(lambda x, y: max(x,y))
    # wi_avg = w.map(lambda x: x[1]).mean()

    # hj_min = h.map(lambda x: x[1]).reduce(lambda x, y: min(x,y))
    # hj_max = h.map(lambda x: x[1]).reduce(lambda x, y: max(x,y))
    # hj_avg = h.map(lambda x: x[1]).mean()

    # print " ------------------ "
    # print " ------------------ "
    # print " ------------------ "
    # print " ------------------ "
    # print " ------------------ "


    # print "vi range:", vi_min, "~", vi_max, "   avg :", vi_avg
    # print "vj range:", vj_min, "~", vj_max, "   avg :", vj_avg

    # print "wi range:", wi_min, "~", wi_max, "   avg :", wi_avg
    # print "hj range:", hj_min, "~", hj_max, "   avg :", hj_avg

    # print " ------------------ "
    # print " ------------------ "
    # print " ------------------ "
    # print " ------------------ "
    # print " ------------------ "



    # partition v
    v = v.keyBy(v_key).partitionBy(d)
    v.cache()

    # main mf loop
    for epoch in xrange(num_iterations):

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

                block_i = None

                _vs = None
                _ws = None
                _hs = None

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


                # print "w_is:", w_dict.keys()
                # print "h_js:", h_dict.keys()
                # print "len(vs):", len(_vs)
                # if len(w_dict) == 0:
                #     print "erorr: block_i=", block_i


                for _v in _vs:
                    i, j, r = _v[1], _v[2], _v[3]

                    # print "v: i, j" , i, j
                    # block_i = gen_block_i(i, N, d)
                    block_j = gen_block_j(j, M, d, d_i)
                    if block_i != block_j:
                        continue

                    epsilon = math.pow((100 + n), - beta_value)
                    n += 1

                    esti = float(r) - np.inner(w_dict[i], h_dict[j])

                    delta_w = -2.0 * esti * h_dict[j] + 2.0 * ((lambda_value) / Ni[i]) * w_dict[i]
                    delta_h = -2.0 * esti * w_dict[i] + 2.0 * ((lambda_value) / Nj[j]) * h_dict[j]

                    # print "i, j, r", i, j, r
                    # print "before: "
                    # print "w[i]", w_dict[i]
                    # print "h[j]", h_dict[j]
                    # print "Ni[i]", Ni[i], "Nj[j]", Nj[j]
                    # print "esti", esti
                    # print "delta_w", delta_w
                    # print "delta_h", delta_h
                    # print "epsilon", epsilon

                    w_dict[i] = w_dict[i] - epsilon * delta_w
                    h_dict[j] = h_dict[j] - epsilon * delta_h

                    # print "after: "
                    # print "w[i]", w_dict[i]
                    # print "h[j]", h_dict[j]

                ret = []
                for i, vector in w_dict.items():
                    ret.append(('w', i, vector, Ni[i]))
                for j, vector in h_dict.items():
                    ret.append(('h', j, vector, Nj[j]))

                return ret



            new_w_h = v.groupWith(
                        w.keyBy(w_key).partitionBy(d),
                        h.keyBy(h_key).partitionBy(d)
                    ).mapPartitions(sgd_partitions)
            new_w_h.cache()

            w = new_w_h.filter(lambda x: x[0] == 'w')
            h = new_w_h.filter(lambda x: x[0] == 'h')

        if IS_TEST:
            dump_WH(test_output_path, w, h, N, M, num_factors, epoch)

    dump_W(output_w_filepath, w, N, num_factors)
    dump_H(output_h_filepath, h, M, num_factors)


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
