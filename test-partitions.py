from pyspark import SparkContext
import numpy as np
from random import randint

sc = SparkContext("local", "DSGD")

# generate w
w = sc.parallelize(
    [(i, np.array([randint(0, 10) for k in range(10)]))
        for i in range(100)]
)

w = sc.parallelize(
    [
        ('w', 0, np.array([3])), ('w', 1, np.array([4])),
        ('h', 0, np.array([3])), ('h', 1, np.array([4])),
        ('v', 0, 0, 1), ('v', 0, 1, 2), ('v', 1, 0, 3), ('v',1 ,1, 4)
    ]
)


def f(iterator):
    ret = []
    for _ in iterator:
        vs ,ws ,hs = _[1][0], _[1][1], _[1][2]
        for w in ws:
            new_w = (w[0], w[1], w[2]+1)
            ret.append(new_w)
        for h in hs:
            new_h = (h[0], h[1], h[2]+1)
            ret.append(new_h)
        # print _[1]
        # # ret.append(_[1]+1)
        # if _[0] == 'w':
        #     _[2][0] = -1
    return ret 

def printf(iterator):
    for _ in iterator:
        print _

        vs, ws, hs = _[1][0], _[1][1], _[1][2]

        print "W"
        for w in ws:
            print w

        print "H"
        for h in hs:
            print h

    return []
print "....testing...."

v = sc.parallelize([('v', 0, 0, 1), ('v', 0, 1, 2), ('v', 1, 0, 3), ('v',1 ,1, 4)]).keyBy(lambda x: x[1])
h = sc.parallelize([('h', 0, np.array([5])), ('h', 1, np.array([6]))]).keyBy(lambda x: x[1])
w = sc.parallelize([('w', 0, np.array([3])), ('w', 1, np.array([2]))]).keyBy(lambda x: x[1])


v.groupWith(h, w).mapPartitions(printf).collect()

new_w_h = v.groupWith(h, w).mapPartitions(f)
w = new_w_h.filter(lambda x: x[0]=='w').keyBy(lambda x: x[1])
h = new_w_h.filter(lambda x: x[0]=='h').keyBy(lambda x: x[1])
v.groupWith(h, w).mapPartitions(printf).collect()


# partitions = w.map(lambda x: (x[1] % 2, x)).partitionBy(2)

# print partitions.collect()

# y = partitions.mapPartitions(f)

# print partitions.collect()
# print partitions.collect()

print "....end testing...."
