# representation:
#  2s complement, N uint64s as last dimension of ndarray
#  the 2s complement is considered over size N,
#    so the sign bit of the highest index value is the real sign bit
#  the reason to use the last dimension is to align with default broadcasting
#    and make initial implementation easier and clearer.
#  the first dimension might be more computationally efficient and would simply
#    mean permuting axes to broadcast consistently.

class NDBigInt:
    def __init__(self, data, *, _xp=None, copy=None):
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
            self._data = xp.astype(data[...,None], xp.uint64, copy=copy)
            self._limbs = 1
    @property
    def limbs(self):
        return self._limbs
    @property
    def shape(self):
        return self._data.shape[:-1]
    def broadcast_arrays(*arrays):
        xp = arrays[0].xp
        limbs = max([ary.limbs for ary in arrays])
        [ary._reserve(limbs) for ary in arrays]
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

        # it might be possible to vectorize this more, but it seems a little
        # complex. each approach i found still involved treating the data in
        # groups, simply larger ones, so i just wrote the group code to start

        xp = x.xp
        size = x._data.shape[axis]
        copy = True
        slices = [slice(None)] * axis + [None,...]
        slices[axis] = slice(0,None,2); slices_A = slices
        slices[axis] = slice(0,-1,2); slices_A_one_less = slices
        slices[axis] = slice(1,None,2); slices_B = slices
        while size > 1:
            y = x[slices_B]
            x = NDBigInt(x._data[slices_A], copy=copy)
            copy = False
            if size % 2:
                x[slices_A_one_less] += y
            else:
                x[slices_A] += y
            size = x._data.shape[axis]
        if not keepdims:
            slices[axis] = 0
            return x[slices]
        else:
            return x
    def __iadd__(x, y):
        # copy of __isub__ with -= changed to +=

        xp = x.xp
        limbs = max(x.limbs, y.limbs)
        alloc = limbs + 1
        x._reserve(alloc)
        y._reserve(alloc)
        y_limbs = y._data[...,:limbs]
        x_limbs = x._data[...,:limbs]
        x._data[...,:limbs] += y_limbs
        # in cases of overflow, the sum is less than the addend
        # if a value all 0xf, as for negative numbers, there will be multiple chained overflows
        raise NotImplementedError('fix logic in carry code')
        while True:
            oflows = x._data[...,:limbs-1] < y_limbs
            if not xp.any(oflows):
                break
            x._data[...,1:limbs] += xp.astype(oflows, xp.uint8, copy=False)
        # if the last bit overflows, we don't want to change the sign of the result, and need to increase the limbs.
        end_oflows = x._data[...,limbs-1] < y_limbs[...,-1]
        if xp.any(end_oflows):
            x._data[...,limbs] |= xp.astype(end_oflows, xp.uint8, copy=False)
            limbs = alloc
        x._limbs = limbs
        return x
    def __isub__(x, y):
        # copy of __iadd__ with += changed to -=

        xp = x.xp
        limbs = max(x.limbs, y.limbs)
        alloc = limbs + 1
        x._reserve(alloc)
        y._reserve(alloc)
        y_limbs = y._data[...,:limbs]
        x_limbs = x._data[...,:limbs]
        x._data[...,:limbs] -= y_limbs
        # in cases of overflow, the sum is less than the addend
        # if a value all 0xf, as for negative numbers, there will be multiple chained overflows
        while True:
            oflows = x._data[...,:limbs-1] < y_limbs[...,:-1]
            if not xp.any(oflows):
                break
            x._data[...,1:limbs] -= xp.astype(oflows, xp.uint8, copy=False)
        # if the last bit overflows, we don't want to change the sign of the result, and need to increase the limbs.
        end_oflows = x._data[...,limbs-1] < y_limbs[...,-1]
        if xp.any(end_oflows):
            x._data[...,limbs] |= xp.astype(end_oflows, xp.uint8, copy=False)
            limbs = alloc
        x._limbs = limbs
        return x
    def __eq__(x, y):
        xp = x.xp
        return xp.all(x._data == y._data, axis=-1)
    def _reserve(self, limbs):
        old_limbs = self.limbs
        if old_limbs < limbs:
            xp = self.xp
            new_data = xp.empty([*self._data.shape[:-1], limbs], dtype=xp.uint64)
            new_data[...,:old_limbs] = self._data[...,:old_limbs]
            # sign extension
            #new_data[old_limbs:,self._data[-1,...]>=UINT64_SIGN] = UINT64_MAX
            new_data[...,old_limbs:] = (xp.astype(self._data[...,old_limbs-1], xp.int64, copy=False) >> 63)[...,None]
            self._data = new_data
    def __int__(self):
        xp = self.xp
        accum = int(xp.astype(self._data[...,-1], xp.int64, copy=False))
        for item in self.xp.unstack(self._data[...,:-1][...,::-1]):
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
    import numpy as np
    np.random.seed(0)
    ars = [
        NDBigInt(xp.asarray(np.random.randint(0,1<<64,[64,64,64], dtype=np.uint64)))
        for idx in range(64)
    ]
    ar = NDBigInt(ars[0], copy=True)

    simple_test_ndbi = NDBigInt(ar[0,0,0], copy=True)
    simple_test_i = int(simple_test_ndbi)
    simple_test_ndbi += simple_test_ndbi
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


# notes on addition error
# - the initial summation produces the correct full result
#   using sign extension without any carry
# - however, because the numbers are negative, of course there is carry
#   when they are treated as unsigned.
# - what is not detected is that this makes the final byte 0, overflowwing
#   into any interpretation of sign flag

# so that's notable with both x and y here. additionally, i suspect
#  this case can only happen when both values are negative

# so if the two addends are both negative, and the sum is not negative,
# then .... uh ... the final carry bit shouldn't be added i guess?
# at ... what point?
# i guess at the end of the top limb

# it realtes to the final oflow
# but this could also relate to the oher oflows accumulating
# if we consider in general blocks of 0xf summing

# this problem is kind of a special case of not accumulating 0xf blocks

# 0x06 0xff 0xff 0x06
# 0x01 0x00 0x00 0xfa
# -------------------
# 0x07         1
# 0x07    1
# 0x08 0x00 0x00 0x00

# this would happen mostly only for negative numbers
# and occasionally for positive
# and would usually overflow as it's a negative number

# so the theory is something like
# (a) it would need to check for overflow after carry again
# (b) if the sign bit is overflowed into, it might be a special case
#     in that a negative sign bit should stay negative

# there's maybe a remaining concern that small negative numbers
# in tensors with many limbs then always hit the worst case maximum
# number of repeated iterations

# # one idea is to consider using a negative representation somewhere
# # it's a += / -= operation
# given that l
