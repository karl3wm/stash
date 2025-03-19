import numpy as np
from math import gcd
from timeit import timeit

def numpy_gcd(a, b):
    a, b = np.broadcast_arrays(a, b)
    a = a.copy()
    b = b.copy()
    pos = np.nonzero(b)[0]
    while len(pos) > 0:
        b2 = b[pos]
        a[pos], b[pos] = b2, a[pos] % b2
        pos = pos[b[pos]!=0]
    return a
#Here is the code to test the result and speed:
#
#In [181]:
n = 2000
a = np.random.randint(100, 1000, n)
b = np.random.randint(1, 100, n)
al = a.tolist()
bl = b.tolist()
cl = zip(al, bl)
#from fractions import gcd
#g1 = numpy_gcd(a, b)
#g2 = [gcd(x, y) for x, y in cl]
#print(np.all(g1 == g2))
#
#True
#
#In [182]:
print('numpy_gcd', timeit('numpy_gcd(a, b)', globals=locals(), number=1000))
#
#1000 loops, best of 3: 721 us per loop
#
#In [183]:
print('gcd', timeit('[gcd(x, y) for x, y in cl]', globals=locals(), number=1000))
#
#1000 loops, best of 3: 1.64 ms per loop
