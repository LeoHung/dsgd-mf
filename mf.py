from sys import argv
import numpy as np
from random import random, gauss
import math
import os
from datetime import datetime


def read_input(input_v_filepath):
    f = open(input_v_filepath)
    ret = []
    for line in f:
        tmp = line.split(",")
        i, j, r = int(tmp[0]) - 1, int(tmp[1]) - 1, int(tmp[2])
        ret.append((i, j, r))
    f.close()
    return ret


def new_dump_W(filename, w, N, num_factors):
    start_time = datetime.now()

    w_matrix = np.zeros((N, num_factors))
    for i, w_vector in w.items():
        w_matrix[i] = w_vector

    np.savetxt(filename, w_matrix, fmt='%.4f', delimiter=",")

    print "dump_W : %f secs" % (datetime.now() - start_time).total_seconds()


def new_dump_H(filename, h, M, num_factors):
    start_time = datetime.now()

    h_matrix = np.zeros((M, num_factors))
    for j, h_vector in h.items():
        h_matrix[j] = h_vector

    np.savetxt(filename, h_matrix.T, fmt='%.4f', delimiter=",")

    print "dump_H : %f secs" % (datetime.now() - start_time).total_seconds()



def sum_lnzsl(v, w, h):
    lnzsl = 0.0
    for i, j, r in v:
        lnzsl += (r - np.inner(w[i], h[j])) ** 2
    return lnzsl


def rmse(total_lnzsl, total_v):
    from math import sqrt
    return sqrt(total_lnzsl / total_v)


def main(
        num_factors, num_workers, num_iterations, beta_value, lambda_value,
        input_v_filepath, output_w_filepath, output_h_filepath):

    v = read_input(input_v_filepath)

    total_v = len(v)

    N = max(map(lambda x: x[0], v))
    M = max(map(lambda x: x[1], v))

    Ni = {}
    Nj = {}
    for i, j, r in v:
        if i not in Ni:
            Ni[i] = 0
        if j not in Nj:
            Nj[j] = 0
        Ni[i] += 1
        Nj[j] += 1

    w = {}
    for i, j, r in v:
        if i not in w:
            w[i] = np.array([0.01 * gauss(0, 1) for _ in xrange(num_factors)])

    h = {}
    for i, j, r in v:
        if j not in h:
            h[j] = np.array([0.01 * gauss(0, 1) for _ in xrange(num_factors)])

    all_lnzsl = []
    all_rmse = []

    n = 0
    for epoch in xrange(num_iterations):
        print " --------- EPOCH %d --------- " % epoch
        start_time = datetime.now()

        for i, j, r in v:
            epsilon = math.pow((100 + n), - beta_value)
            esti = float(r) - np.inner(w[i], h[j])

            delta_w = -2.0 * esti * h[j] + 2.0 * ((lambda_value) / Ni[i]) * w[i]
            delta_h = -2.0 * esti * w[i] + 2.0 * ((lambda_value) / Nj[j]) * h[j]

            w[i] = w[i] - epsilon * delta_w
            h[j] = h[j] - epsilon * delta_h

            n += 1

        lnzsl = sum_lnzsl(v, w, h)
        all_lnzsl.append(lnzsl)
        all_rmse.append(rmse(lnzsl, total_v))

        print "-- Epoch %d: lnzsl[%f], rmse[%f]" % (epoch, all_lnzsl[-1], all_rmse[-1])
        print "-- time: %f " % (datetime.now() - start_time).total_seconds()

    print "i, total_lnzsl"
    for i, total_lnzsl in enumerate(all_lnzsl):
        print "%d,%f" % (i, total_lnzsl)

    print "--------------"
    print "i, rmse"
    for i, rmse_value in enumerate(all_rmse):
        print "%d,%f" % (i, rmse_value)

    # output
    new_dump_W(output_w_filepath, w, N, num_factors)
    new_dump_H(output_h_filepath, h, M, num_factors)


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