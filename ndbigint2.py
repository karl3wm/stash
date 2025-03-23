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
    astore = a.storage()
    bstore = b.storage()
    astart = astore.data_ptr()
    bstart = bstore.data_ptr()
    aend = astart + astore.nbytes()
    bend = bstart + bstore.nbytes()
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
    def __init__(self, data, *, _xp=None, copy=None, alloc=None):
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
            if xp.isdtype(data.dtype, xp.uint64) and xp.any(data >= 0x8000000000000000):
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
        # if the end limb overflows then another is needed

        # the test for overflow differs in the last limb, which must be treated as a signed rather than unsigned number.
        # this is because two large positive numbers can overflow into the sign bit, and this is still an overflow, but makes a greater unsigned value
        # one approach would be to use a mask.
        # notably the sign bit is always the highest bit of the last limb.
        # the result is even correct: it is simply interpreted as a negative number.
        # if it were treated as an overflow the code would continue correctly.
        # there is likely a masked xor operation that would make the current code work.

        # the overflow tests looks for sums that are less than addends.
        # but certain large positive numbers overflow into the sign bit.
        # this only happens when both initial numbers are positive (as otherwise the magnitude decreases), and the final number is negative.
        # this sign bit is also set when the lost initial addend is negative and of greater magnitude than y, in which case the sum is less negative.

        # a resource is that both numbers have been sign extended to an entire empty limb.
        # this likely was expected to address situations like this.
        # so we would have the original sign bit preserved if the summation didn't include that final empty limb.

        # maybe i'll take a step here to address the sign of the result
        # there are two possible overflows: a negative number can overflow into the last limb, and a positive number can overflow into the sign bit.
        # i think these both only happen when the original sign bits are identical
        # for negative overflow, two set sign bits become a value of lower total magnitude
        #   no overflow
        #         1  11  1
        #           [11] 11
        #         + [11] 11
        #        -----------
        #            11  10
        #   overflow
        #          1 11
        #           [11] 11
        #         + [11] 10
        #        -----------
        #            11  01
        #   overflow by 2
        #         1  11
        #           [11] 10
        #         + [11] 10
        #        -----------
        #            11  00
        #   the sign bit overflows into the additional limb.
        # for positive overflow, two unset sign bits become set
        #   overflow
        #           [00] 01
        #         + [00] 01
        #        -----------
        #           [00] 10
        #   now, like with negative, the additional limb is needed to preserve the sign.

        #   in both conditions of overflow, the additional limb's original sign is what is valuable.
        #   it could be kept as-is with sign extension

        #   meanwhile, when a negative and a positive number are summed, the resulting sign relates to the addend of larger magnitude
        #   it could be re-sign-extended from the portions summed.

        # so, possibly two conditions: overflow, and non-overflow.
        # for overflow, we want our original sign.
        # for non-overflow, we want to sign extend our new sign bit.
        # overflow can be detected by (a) the addend sign bits matching and (b) the sum sign bit mismatching
        # this could be simplified into: if the addend sign bits match, then use the original limb
        # if the addend sign bits mismatch, then sign-extend

        #if xp.any(x._data[...,limbs-1] < y._data[...,limbs-1]):
        #    limbs = alloc
        #elif xp.any((x._data[...,limbs] == 0) & (y._data[...,limbs] == 0) & (x._data[...,limbs-1] > 0x8000000000000000)):
        #    # a positive summation overflowed into the sign bit
        #    limbs = alloc
        # if a limb is all 0xf, as for negative numbers, there will be multiple chained overflows
        # one way to reduce the iterations here would be to amortize over many operations by maintaining overflow data, looping until it is 0 when __int__ is called
        ref = y._data[...,:limbs]
        off = 0
        while True:
            oflows = x._data[...,off:limbs-1] < ref[...,:-1] # this also does not detect when the final limb has overflowed into the sign bit
            if not xp.any(oflows):
                break
            ref = xp.astype(oflows, xp.uint8, copy=False)
            off += 1
            x._data[...,off:limbs] += ref

        # we want to sign extend only the numbers that have mismatching addend sign bits
        # that is, the numbers where the new sign mismatches that in the additional limb

        # how to calculate that. 3 parts .. y's top 65 bits, x's top 64 bits, and x's 65th bit the new sign bit
        # - if bit 65 == bits 0-64, leave them as is
        # - if bit 65 != bits 0-64, sign extend. if so, isn't that the same as inverting?
        #   oh the new bit differs from the old. 3 values.
        # - if y's bit -65 == x bits -64:, leave them as is
        # - if y's bit -65 != x bits -64:, then sign extend x's bit -65 into bits -64:
        # the problem is simplifiable into roughly conditioning the sign extend.
        # - we can calculate the sign extension content by shifting bit -65
        # - then we place it only if the sign mismatches
        # we'd likely do the conditioning prior to the extending, might not matter
        # so, what bit will we fill the end with?
        # find boolean expression for this bit.
        # - ybits[-65] == xbits[-1] -> xbits[-1] (or equivalently ybits[-65])
        # - ybits[-65] != xbits[-1] -> xbits[-65]
        # ysign x0sign  choice
        # 0     0       0
        # 0     1       x1sign
        # 1     0       x1sign
        # 1     1       1

        # hrm
        # ((ysign ^ x0sign) & x1sign) | ((~ysign ^ x0sign) & ysign)
        # ~(~((ysign ^ x0sign) & x1sign) & ~(~(ysign ^ x0sign) & ysign))

        # ysign x0sign  x1sign choice
        # 0     0       0      0
        # 0     0       1      0
        # 0     1       0      0
        # 0     1       1      1
        # 1     0       0      0
        # 1     0       1      1
        # 1     1       0      1
        # 1     1       1      1

        # expanded table makes clearer that choice is ysign unless x0sign == x1sign != ysign
        # so something like ysign ^ (x0sign ^ x1sign) with some nots?

        # ysign x0sign  x1sign choice x0^x1
        # 0     0       0      0      0
        # 0     0       1      0      1
        # 0     1       0      0      1
        # 0     1       1      1      0
        # 1     0       0      0      0
        # 1     0       1      1      1
        # 1     1       0      1      1
        # 1     1       1      1      0

        # oops it's ysign unless x0sign != ysign and x0sign == x1sign

        # ysign x0sign  x1sign choice x0^x1 y^x0 y^x1  (y^x0)&(y^x1) y^((y^x0)&(y^x1))
        # 0     0       0      0      0     0    0     0             0
        # 0     0       1      0      1     0    1     0             0
        # 0     1       0      0      1     1    0     0             0
        # 0     1       1      1      0     1    1     1             1
        # 1     0       0      0      0     1    1     1             0
        # 1     0       1      1      1     1    0     0             1
        # 1     1       0      1      1     0    1     0             1
        # 1     1       1      1      0     0    0     0             1

        # well that's 4 operations on a 1 bit value
        # involving 2 temporaries that could be put into 1 word
        # so the extra limb is initially filled with x0.
        # could maybe place x1 into it somehow .. hrm
        # if i could make part of it be x1 .. then ...
        # this seems like too many operations. there must be a better approach. but i can try this one.

        # do any existing things include this?
        # - the carry code xors x1 with x0 when there is overflow if let to write to the extra limb
        # if we continued the concept of 'overflow', one could identify overflow by comparing the sign changes in sequence
        # x0sign == ysign && x1sign != x0sign
        # so the logical == operator is useful because it's ~^. so there are more operations available if working with bool.

        # ysign x0sign  x1sign choice x0^x1 y^x0 y^x1  y^x0^x1  x0==x1  y==x0  y==x1   ((x0==x1)&(y^x0))^y
        # 0     0       0      0      0     0    0     0        1       1      1
        # 0     0       1      0      1     0    1     1        0       1      0
        # 0     1       0      0.     1     1    0     1        0       0      1
        # 0     1       1      1      0     1    1     0        1       0      0
        # 1     0       0      0      0     1    1     1        1       0      0
        # 1     0       1      1.     1     1    0     0        0       0      1
        # 1     1       0      1      1     0    1     0        0       1      0
        # 1     1       1      1      0     0    0     1        1       1      1

        ## so we could make a bool tensor from x0, then compare it with x1, maybe ...
        #x0_sign = xp.astype(x._data[...,limbs], dtype=xp.bool, copy=False)
        #y_sign = xp.astype(y._data[...,limbs], dtype=xp.bool, copy=False)
        #result_sign = y_sign ^ x0_sign
        #x1_sign = x._data[...,limbs-1] >= 0x8000000000000000
        #result_sign &= (x0_sign == x1_sign)
        #result_sign ^= y_sign
        ## might be convenient to work in full blocks; only x1 is unavailable in a full block. but it could be sign extended
        ## but one could instead subtract a bool from 0 to turn it into a full block
        ## it could be convenient if the result sign were in terms of x0 sign which is already the full block

        # x0sign ^ ((x0^x1)&(y==x1))

        # might not help too much

        # note i was earlier using signed >> 63 to convert 1 bit to 64 bits
        x0_sign = xp.astype(x._data[...,limbs], xp.bool, copy=False)
        y_sign = xp.astype(y._data[...,limbs], xp.bool, copy=False)
        result_sign = y_sign ^ x0_sign
        x1_sign = x._data[...,limbs-1] >= 0x8000000000000000
        result_sign &= (x0_sign == x1_sign)
        result_sign ^= y_sign
        x._data[...,limbs:] = xp.astype(-xp.astype(result_sign[...,None], xp.int64, copy=False), xp.uint64, copy=False)

        # sign extend, set limbs
        # probably efficiency improvements exist

        if xp.any((x._data[...,limbs]) ^ (x._data[...,limbs-1]>>63)):
            limbs += 1
        else:
            signed_x = xp.astype(x._data, xp.int64, copy=False)
            #x._data[...,limbs:] = xp.astype(signed_x[...,limbs-1:limbs] >> 63, xp.uint64, copy=False)
            while limbs > 1 and xp.all(signed_x[...,limbs-1] == signed_x[...,limbs-2]>>63):
                limbs -= 1
        x._limbs = limbs
        #actual_sum = x._tolist(); assert expected_sum == actual_sum
        return x
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
    def _alloc(self, alloc):
        old_alloc = self._data.shape[-1]
        if old_alloc < alloc:
            new_data = self.xp.empty([*self._data.shape[:-1], alloc], dtype=xp.uint64)
            new_data[...,:old_alloc] = self._data[...,:old_alloc]
            self._data = new_data
        else:
            assert self._limbs <= alloc
            #self._data = self._data[...,:alloc]
        if alloc > self._limbs:
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
