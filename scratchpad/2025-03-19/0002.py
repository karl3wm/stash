class NDFraction:
    def __init__(self, numerator, denominator=1):
        xp = self.xp = numerator.__array_namespace__()
        assert 'complex' not in str(numerator.dtype)
        if 'float' in str(numerator.dtype):
            self._num_den = xp.asarray([
                float(v).as_integer_ratio()
                for v in xp.unstack(n)
            ]).T.view(2,*numerator.shape)
            self /= denominator
        else:
            self._num_den = xp.broadcast_arrays(numerator, denominator)
            self.simplify()
        self.shape = list(self._num_den.shape[1:])

    @property
    def numerator(a):
        return self._num_den[0]
    @property
    def denominator(a):
        return self._num_den[1]

    def simplify(a):
        a = a._num_den
        xp = a.__array_namespace__()
        a /= gcd(xp, a[0], a[1])

    def __add__(a, b):
        a = a._num_den
        b = b._num_den
        xp = a.__array_namespace__()
        g = gcd(a[1,...], b[1,...])
        #if (g == 1).all():
        #    a[0


def igcd(xp, b, a):
    a_mod_b_ = a % b
    a = b
    return a
    if xp.any(a_mod_b_):
        b = a_mod_b_
        mask = a_mod_b_ != 0
        b_ = a_mod_b_[mask]
        a_mod_b_ = a[mask] % b_
        a[mask] = b_
        while xp.any(a_mod_b_):
            b[mask] = a_mod_b_
            mask_ = a_mod_b_ != 0
            mask[mask] = mask_
            a_ = b_[mask_]
            b_ = a_mod_b_[mask_]
            a_mod_b_ = a_ % b_
            a[mask] = b_
    return a

if __name__ == '__main__':
    #import array_api_strict as xp
    #import math, random
#
#    print(igcd(xp, xp.asarray([[12,256],[6,256]]), xp.asarray([[9,480],[15,8]])))

    import numpy as np
    import timeit
    from math import gcd as math_gcd
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
    n = 2000*1000
    a = np.random.randint(100, 1000, n)
    b = np.random.randint(1, 100, n)
    al = a.tolist()
    bl = b.tolist()
    cl = zip(al, bl)
    g2 = [math_gcd(x, y) for x, y in cl]
    g1 = igcd(np, a, b)
    print(np.all(g1 == g2))
    print('math_gcd', timeit.timeit('[math_gcd(x, y) for x, y in cl]', globals=locals(),number=1))
    #print('numpy_gcd', timeit.timeit('numpy_gcd(a, b)', globals=locals()))
    print('igcd', timeit.timeit('igcd(np, a, b)', globals=locals(), number=1))
