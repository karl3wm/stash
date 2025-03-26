# representation:
#  2s complement, N uint64s as last dimension of ndarray
#  the 2s complement is considered over size N,
#    so the sign bit of the highest index value is the real sign bit
#  the reason to use the last dimension is to align with default broadcasting
#    and make initial implementation easier and clearer.
#  the first dimension might be more computationally efficient and would simply
#    mean permuting axes to broadcast consistently.

def may_share_memory_torch(a, b):
    if a.device != b.device:
        return False
    astart = a.data_ptr()
    bstart = a.data_ptr()
    astride = a.stride()
    bstride = b.stride()
    astride, adim = max([[astride[idx],idx] for idx in range(len(astride))])
    bstride, bdim = max([[bstride[idx],idx] for idx in range(len(bstride))])
    aend = astart + a.size(adim) * astride * a.element_size()
    bend = bstart + b.size(bdim) * bstride * b.element_size()
    return aend > bstart and bend > astart

def may_share_memory_numpy(a, b):
    import numpy as np
    return np.may_share_memory(a, b)

def _may_share_memory(xp, a, b):
    try:
        return may_share_memory_numpy(a._array, b._array)
    except AttributeError as e:
        raise Exception("implement _may_share_memory", e)


class NDBigInt:
    def __init__(self, data, *, copy=None, alloc=None, _xp=None, _limbs=None):
        if type(data) is NDBigInt:
            xp = self.xp = data.xp
            self._data = xp.asarray(data._data, copy=copy)
            self._limbs = data._limbs
        else:
            if _xp is None:
                _xp = data.__array_namespace__()
            xp = self.xp = _xp
            if not xp.isdtype(data.dtype, 'integral'):
                raise TypeError(data.dtype)
            if _limbs is not None:
                assert _limbs <= data.shape[-1]
                self._data = data
                self._limbs = _limbs
            elif xp.isdtype(data.dtype, xp.uint64) and xp.any(data >= 0x8000000000000000):
                self._data = xp.empty([*data.shape, 2], dtype=xp.uint64)
                self._data[...,0] = data
                self._data[...,1] = 0
                self._limbs = 2
            else:
                self._data = xp.astype(data[...,None], xp.uint64, copy=bool(copy))
                self._limbs = 1
        if alloc is not None:
            self._alloc(alloc)
    @property
    def limbs(self):
        return self._limbs
    @property
    def shape(self):
        return self._data.shape[:-1]
    def broadcast_arrays(*arrays):
        xp = arrays[0].xp
        limbs = max([ary.limbs for ary in arrays])
        [ary._alloc(limbs) for ary in arrays]
        return [
            NDBigInt(ary, _xp = xp)
            for ary in xp.broadcast_arrays(*[ary._data for ary in arrays])
        ]
    def reshape(x, /, shape, **kwparams):
        x = NDBigInt(x)
        xp = x.xp
        x._data = xp.reshape(x._data, [*shape, x.limbs], **kwparams)
        return x
    def sum(x, *, axis = None, keepdims = False):
        if axis is None:
            return x.reshape([-1]).sum(keepdims = keepdims)

        # i think the immediately clearest way to vectorize this would be to
        # assert that the dimension size fits within 32 bits (<sqrt(64bits))
        # and then sum the high and low 32 bit parts separately each in
        # 64-bit storage.
        # it could generalize to sizes > 32 bits with smaller groups
        # but that seems unneeded here, if this code were ever used for such
        # a gigantic context there would be more devs

        xp = x.xp
        size = x._data.shape[axis]
        copy = True
        preceding_axes = [slice(None)] * (axis - 1)

        while size > 1:
            midsize = size//2
            y = x
            if size % 2:
                x = NDBigInt(y[*preceding_axes,:midsize+1,...], copy=copy)
                x._data[*preceding_axes,midsize,...] = 0
            else:
                x = NDBigInt(y[*preceding_axes,:midsize,...], copy=copy)
            copy = False
            x += y[*preceding_axes,midsize:size,...]
            size = x._data.shape[axis]

        if keepdims:
            return x
        else:
            return x[*preceding_axes,0,...]

    def _tolist(*xs, visitor = None):
        if visitor is None:
            if len(xs) == 1:
                visitor = lambda x: int(x)
            else:
                visitor = lambda *xs: [int(x) for x in xs]
        self = xs[0]
        if len(self.shape):
            return [
                self[idx,...]._tolist(*[x[idx,...] for x in xs[1:]], visitor=visitor)
                for idx in range(self.shape[0])
            ]
        else:
            return visitor(*xs)

    def __iadd__(x, y):
        xp = x.xp
        #x_list = x._tolist()
        #y_list = y._tolist()
        #expected_sum = x._tolist(y, visitor=lambda x, y: int(x) + int(y))
        limbs = max(x.limbs, y.limbs)
        alloc = limbs + 1
        x._alloc(alloc)
        y._alloc(alloc)

        if _may_share_memory(xp, x._data, y._data):
            raise ValueError('edge case: detect overflow for in-place add to self. is a multiply reasonable here?')
            # this is simply to detect overflow!
            # nails might work better for this case
            y = NDBigInt(y, copy=True)

        x._data[...,:limbs] += y._data[...,:limbs]
        # in cases of overflow, the sum is less than the addend

        # if a limb is all 0xf, as for negative numbers, there will be multiple chained overflows
        # one way to reduce the iterations here could be to amortize over many operations by maintaining overflow data, looping until it is 0 when __int__ is called
        ref = y._data[...,:limbs]
        off = 0
        while True:
            oflows = x._data[...,off:limbs-1] < ref[...,:-1] # this also does not detect when the final limb has overflowed into the sign bit
            if not xp.any(oflows):
                break
            ref = xp.astype(oflows, xp.uint8, copy=False)
            off += 1
            x._data[...,off:limbs] += ref

        # sign extend

        # there's likely a way to simplify this.
        # one idea: "why is this needed? what case is addition with sign extension not covering?"
                # this is covering when positive numbers overflow into appearing negative without overflowing their limbs
                # one could also look into expanding the limb incrementation condition instead.
                # although there is also some interest in removing all branching from the function
                    # but wouldn't positive overflow be handled by including the extra limb, to hold the real sign?

        # ysign x0sign  x1sign choice x0^x1 y^x0 y^x1  y^x0^x1  x0==x1  y==x0  y==x1   ((x0==x1)&(y^x0))^y
        # 0     0       0      0      0     0    0     0        1       1      1
        # 0     0       1      0      1     0    1     1        0       1      0
        # 0     1       0      0.     1     1    0     1        0       0      1
        # 0     1       1      1      0     1    1     0        1       0      0
        # 1     0       0      0      0     1    1     1        1       0      0
        # 1     0       1      1.     1     1    0     0        0       0      1
        # 1     1       0      1      1     0    1     0        0       1      0
        # 1     1       1      1      0     0    0     1        1       1      1

        # note i was earlier using signed >> 63 to convert 1 bit to 64 bits
        x0_sign = xp.astype(x._data[...,limbs], xp.bool, copy=False)
        y_sign = xp.astype(y._data[...,limbs], xp.bool, copy=False)
        result_sign = y_sign ^ x0_sign
        x1_sign = x._data[...,limbs-1] >= 0x8000000000000000
        result_sign &= (x0_sign == x1_sign)
        result_sign ^= y_sign
        x._data[...,limbs:] = xp.astype(-xp.astype(result_sign[...,None], xp.int64, copy=False), xp.uint64, copy=False)
        #x._data[...,limbs:] = xp.astype(signed_x[...,limbs-1:limbs] >> 63, xp.uint64, copy=False)

        # i want to compare only the 63rd bit
        # i'm interested in (a&63rd)==(b&63rd)
        # which i guess is ~(a&63rd)^(b&63rd)
        # given xor is bitwise
        # we can do (a^b)&63rd == 0

        # set limbs
        # probably efficiency improvements exist
        if xp.any((x._data[...,limbs] ^ x._data[...,limbs-1]) & 0x80000000_00000000):
            limbs += 1
        else:
            signed_x = xp.astype(x._data, xp.int64, copy=False)
            while limbs > 1 and xp.all(signed_x[...,limbs-1] == signed_x[...,limbs-2]>>63):
                limbs -= 1
        x._limbs = limbs
        #actual_sum = x._tolist(); assert expected_sum == actual_sum
        return x
    def __mul__(x, y):
        xp = x.xp
        #if _may_share_memory(xp, x._data, y._data):
        #    raise NotImplementedError('in place overlapping multiply')

        # I considered this as whole words at first, then implemented it with
        # halflimbs as two separate blocks.  However, they are likely unifiable
        # with a slightly different matrix reshaping.
        # This would give the intermediate values a more intuitive arrangement
        # during debugging or future changes.
        # The arithmetic may also be incorrect until these are considered together.
        # Additionally the two matmuls currently present could possibly be
        # unified into one somehow.

        xlimbs = x._limbs
        ylimbs = y._limbs
        x = x._data
        y = y._data
        shape = x.shape[:-1]

        # = sum([
        #   [x[0]*y[0],         0,         0],
        #   [x[0]*y[1], x[1]*y[0],         0],
        #   [x[0]*y[2], x[1]*y[1], x[2]*y[0]],
        #   [        0, x[1]*y[2], x[2]*y[1]],
        #   [        0,         0, x[2]*y[2]],
        # ], axis=-1)

        # = sum([
        #   [x[0]*y[0], x[0]*y[1], x[0]*y[2],         0,         0],
        #   [        0, x[1]*y[0], x[1]*y[1], x[1]*y[2],         0],
        #   [        0,         0, x[2]*y[0], x[2]*y[1], x[2]*y[2]],
        # ], axis=-2)

        final_limbs = (ylimbs + 1) * 2

        prod = xp.empty(
            [*shape, 2, xlimbs, final_limbs + 1],
            dtype = xp.uint64
        )

        # construct the outer products of halflimbs masked to collect overflow
        # and carry information and padded with zeros

        # low halflimbs can be multiplied in-place
        prod[..., 0, :, :ylimbs] = (
                (x[...,:xlimbs,None] & 0x00000000ffffffff)
                @
                (y[...,None,:ylimbs] & 0x00000000ffffffff)
        )
        prod[..., 0, :, ylimbs:] = 0

        # high halflimbs have an extra 1<<32 factor and end up 1 limb higher
        # than they started
        prod[..., 1, :, 0] = 0
        prod[..., 1, :, 1:ylimbs+1] = (
                (x[...,:xlimbs,None] >> 32)
                @
                (y[...,None,:ylimbs] >> 32)
        )
        prod[..., 1, :, ylimbs+1:] = 0

        # reshape with the padded dimension 1 size smaller (final_limbs)
        # to give the outer product the slided offsetting for the sum
        prod = xp.reshape(
            xp.reshape(
                prod,
                [*shape, 2, -1]
            )[..., :2, :xlimbs * final_limbs],
            [*shape, 2 * xlimbs, final_limbs]
        )

        # then the product might be a bigint sum of prod along -2

        prod = NDBigInt(prod, _xp=xp, _limbs=final_limbs)
        raise NotImplementedError()
        return prod.sum(axis=-2)

    def __isub__(x, y):
        x._data ^= 0xffffffffffffffff
        x += y
        x._data ^= 0xffffffffffffffff
        return x
    def __add__(x, y):
        x = NDBigInt(x, copy=True)
        x += y
        return x
    def __sub__(x, y):
        x = NDBigInt(x, copy=True)
        x -= y
        return x
    def __eq__(x, y):
        return x.xp.all(x._data[...,:x._limbs] == y._data[...,:y._limbs], axis=-1)
    def __ne__(x, y):
        return x.xp.any(x._data[...,:x._limbs] != y._data[...,:y._limbs], axis=-1)
    def _alloc(self, alloc, _sign_extend=True):
        old_alloc = self._data.shape[-1]
        if old_alloc < alloc:
            new_data = self.xp.empty([*self._data.shape[:-1], alloc], dtype=xp.uint64)
            new_data[...,:old_alloc] = self._data[...,:old_alloc]
            self._data = new_data
        else:
            assert self._limbs <= alloc
            #self._data = self._data[...,:alloc]
        if alloc > self._limbs and _sign_extend:
            # sign extension
            #new_data[self._data[...,-1]>=UINT64_SIGN,old_alloc:] = UINT64_MAX
            self._data[...,old_alloc:] = self.xp.astype(self._data[...,old_alloc-1,None], xp.int64, copy=False) >> 63
    def __int__(self):
        xp = self.xp
        signlimb = self.limbs - 1
        accum = int(xp.astype(self._data[...,signlimb], xp.int64, copy=False))
        for item in xp.unstack(self._data[...,:signlimb][...,::-1]):
            accum <<= 64
            accum += int(item)
        return accum
    def __getitem__(self, slices):
        item = NDBigInt(self, copy=False)
        if type(slices) is tuple:
            item._data = item._data[*slices,:]
        else:
            item._data = item._data[slices,:]
        return item
    def __str__(self):
        xp = self.xp
        shape = self._data.shape
        idx = [0] * (len(shape) - 2)
        off = len(idx) - 1
        depth = 0
        result = 'NDBigInt: '
        indent = ' ' * len(result)
        #result += '[' * len(idx)
        result += '\n'
        if len(shape) > 2:
            rowidcs = [[idx] for idx in range(shape[-3])]
        else:
            rowidcs = [[]]
        while True:
            for rowidx in rowidcs:
                row = self[*idx, *rowidx, :]
                row = [str(int(row[i])) for i in range(shape[-2])]
                result += indent + '[' + ',\t'.join(row) + ']\n'
            while True:
                if off == -1:
                    return result
                idx[off] += 1
                if idx[off] < shape[off]:
                    break
                idx[off] = 0
                off -= 1

if __name__ == '__main__':
    import array_api_strict as xp

    assert int(NDBigInt(xp.asarray(-6532100632237123854),alloc=3) + NDBigInt(xp.asarray(7958265450555812818),alloc=2)) == 1426164818318688964
    assert int(NDBigInt(xp.asarray(7692698082559361259)) + NDBigInt(xp.asarray(7692698082559361259))) == 15385396165118722518
    assert int(NDBigInt(xp.asarray(15761168082059424201)) + NDBigInt(xp.asarray(8937115293130262283))) == 24698283375189686484

    assert int(NDBigInt(xp.asarray(3)) * NDBigInt(xp.asarray(4))) == 12
    assert int(NDBigInt(xp.asarray(7692698082559361259)) * NDBigInt(xp.asarray(7692698082559361259))) == 59177603789412473292821695494070065081
    assert int(NDBigInt(xp.asarray(15761168082059424201)) * NDBigInt(xp.asarray(8937115293130262283))) == 140859376303769844698647197831087710883
    assert int(NDBigInt(xp.asarray(-3)) * NDBigInt(xp.asarray(4))) == -12
    assert int(NDBigInt(xp.asarray(3)) * NDBigInt(xp.asarray(-4))) == -12
    assert int(NDBigInt(xp.asarray(-3)) * NDBigInt(xp.asarray(-4))) == 12
    assert int(NDBigInt(xp.asarray(-6532100632237123854),alloc=3) * NDBigInt(xp.asarray(7958265450555812818),alloc=2)) == -51984190781086484234522341753506760572

    import numpy as np
    np.random.seed(0)
    ars = [
        NDBigInt(xp.asarray(np.random.randint(0,1<<64,[63,65,67], dtype=np.uint64)))
        for idx in range(3)
    ]
    ar = NDBigInt(ars[0], copy=True)

    simple_test_ndbi = NDBigInt(ar[0,0,0], copy=True)
    simple_test_i = int(simple_test_ndbi)
    simple_test_ndbi += NDBigInt(simple_test_ndbi, copy=True)
    assert int(simple_test_ndbi) == simple_test_i * 2

    a0000 = int(ars[0][0,0,0])
    a1000 = int(ars[1][0,0,0])
    ar += ars[1]
    assert int(ars[0][0,0,0]) == a0000
    assert int(ars[1][0,0,0]) == a1000
    assert int(ar[0,0,0]) == a0000 + a1000
    ar -= ars[1]
    assert int(ar[0,0,0]) == int(ars[0][0,0,0])
    assert xp.all(ar == ars[0])
    ar += ars[2]
    sum1 = ar.sum(axis=0)
    sum2 = ars[0].sum(axis=0)
    assert int(sum2[0,0]) == sum([int(ars[0][idx,0,0]) for idx in range(ars[0].shape[0])])
    sum2 += ars[2].sum(axis=0)
    assert int(sum2[0,0]) == int(sum1[0,0])
    assert xp.all(sum1 == sum2)
