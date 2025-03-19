def test(xp, b, a):
    a_mod_b_ = a % b
    a = b
    return a

if __name__ == '__main__':
    import numpy as np
    import timeit
    from math import gcd as math_gcd
    n = 2000*1000
    a = np.random.randint(100, 1000, n)
    b = np.random.randint(1, 100, n)
    al = a.tolist()
    bl = b.tolist()
    cl = zip(al, bl)
    g2 = [math_gcd(x, y) for x, y in cl]
    g1 = test(np, a, b)
    print('math_gcd', timeit.timeit('[math_gcd(x, y) for x, y in cl]', globals=locals(),number=1))
    print('numpy_mod', timeit.timeit('test(np, a, b)', globals=locals(), number=1))
