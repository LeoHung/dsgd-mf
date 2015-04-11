from sys import argv
from pyspark import SparkContext
import numpy as np
from random import randint
import math


def get_n(data):
    return data.map(lambda x: (0, x[1])).reduce(lambda x, y: max(x, y))[1] + 1


def get_m(data):
    return data.map(lambda x: (0, x[2])).reduce(lambda x, y: max(x, y))[1] + 1


n = 0


def main(
        num_factors, num_workers, num_iterations, beta_value, lambda_value,
        input_v_filepath, output_w_filepath, output_h_filepath):
    sc = SparkContext("local", "DSGD")

    d = num_workers
    # load v
    v = sc.textFile(input_v_filepath)

    def split_data(line):
        tmp = line.split(",")
        return ('v', int(tmp[0]), int(tmp[1]), int(tmp[2]))
    v = v.map(lambda line: split_data(line))
    v.cache()

    total_v = v.count()

    # get m, n
    N, M = get_n(v), get_m(v)

    # get Ni, Nj
    Ni = v.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).map(lambda x: ('Ni', x[0], x[1]))
    Nj = v.map(lambda x: (x[2], 1)).reduceByKey(lambda x, y: x + y).map(lambda x: ('Nj', x[0], x[1]))

    # generate w
    w = sc.parallelize(
        [('w', i, np.array([float(randint(0, 10)) for k in range(num_factors)]))
            for i in range(N)]
    )

    # generate h
    h = sc.parallelize(
        [('h', j, np.array([float(randint(0, 10)) for k in range(num_factors)]))
            for j in range(M)]
    )

    # n = sc.parallelize([('n', 0)])

    all_data = w.union(h).union(v).union(Ni).union(Nj)

    # main mf loop
    for epoch in xrange(num_iterations):
        # generate stratum
        for d_i in xrange(d):
            def v_map(t):
                i, j = t[1], t[2]
                block_i = i / int(math.ceil(float(N) / float(d)))
                block_i = (block_i + d_i) % d
                block_j = j / int(math.ceil(float(M) / float(d)))

                if block_i == block_j:
                    return block_i, t
                else:
                    return d, t

            def w_map(t):
                i = t[1]
                block_i = i / int(math.ceil(float(N) / float(d)))
                block_i = (block_i + d_i) % d
                return block_i, t

            def h_map(t):
                j = t[1]
                block_j = j / int(math.ceil(float(M) / float(d)))
                return block_j, t

            def Ni_map(t):
                return w_map(t)

            def Nj_map(t):
                return h_map(t)

            def stratum_map(t):
                if t[0] == 'w':
                    return w_map(t)
                elif t[0] == 'h':
                    return h_map(t)
                elif t[0] == 'v':
                    return v_map(t)
                elif t[0] == 'Ni':
                    return Ni_map(t)
                elif t[0] == 'Nj':
                    return Nj_map(t)

            def sgd_partitions(iterator):
                w = {}
                h = {}
                Ni = {}
                Nj = {}
                tmp_v = []

                for it in iterator:
                    t = it[1]
                    if t[0] == 'w':
                        i = t[1]
                        w[i] = t[2]
                    elif t[0] == 'h':
                        j = t[1]
                        h[j] = t[2]
                    elif t[0] == 'v':
                        tmp_v.append(t)
                    elif t[0] == 'Ni':
                        Ni[t[1]] = t[2]
                    elif t[0] == 'Nj':
                        Nj[t[1]] = t[2]

                n = epoch * total_v

                if len(w) > 0 and len(h) > 0:
                    for t in tmp_v:
                        i, j, r = t[1], t[2], t[3]
                        wi = w[i]
                        hj = h[j]
                        print "before::"
                        print "i, j, r", i, j, r
                        print "wi:", wi
                        print "hj", hj
                        print "n", n
                        print "Ni", Ni[i], "Nj", Nj[j]

                        n += 1
                        epsilon = math.pow((100 + n), - beta_value)

                        esti = float(r) - np.inner(wi, hj)

                        delta_w = -2.0 * esti * hj + 2.0 * ((lambda_value) / Ni[i]) * wi
                        w[i] -= epsilon * delta_w

                        esti = float(r) - np.inner(wi, hj)

                        delta_h = -2.0 * esti * wi + 2.0 * ((lambda_value) / Nj[j]) * hj
                        h[j] -= epsilon * delta_h

                        print "after::"
                        print "wi", wi
                        print "hj", hj
                        print "n", n

                ret = []
                for i, vector in w.items():
                    ret.append(('w', i, vector))
                for j, vector in h.items():
                    ret.append(('h', j, vector))
                for i, value in Ni.items():
                    ret.append(('Ni', i, value))
                for j, value in Nj.items():
                    ret.append(('Nj', j, value))
                for v in tmp_v:
                    ret.append(v)
                # print ret

                return ret

            all_data = all_data.map(stratum_map).partitionBy(d + 1).mapPartitions(sgd_partitions)

    w_vectors = all_data.filter(lambda x: x[0] == 'w').collect()
    w_vectors = [vector for _, i, vector in sorted(w_vectors, key=lambda x: x[1])]
    w = np.concatenate(w_vectors)
    np.savetxt(output_w_filepath, w)

    h_vectors = all_data.filter(lambda x: x[0] == 'h').collect()
    h_vectors = [vector for _, j, vector in sorted(h_vectors, key=lambda x: x[1])]
    h = np.concatenate(h_vectors)
    np.savetxt(output_h_filepath, h.T)


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
