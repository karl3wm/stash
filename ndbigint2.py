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
    return aend > bstart and bend < astart

def may_share_memory_numpy(a, b):
    import numpy as np
    return np.may_share_memory(a, b)



class NDBigInt:
    def __init__(self, data, *, _xp=None, copy=None):
        if type(data) is NDBigInt:
            xp = self.xp = data.xp
            self._data = xp.asarray(data._data, copy=copy)
            self._limbs = data._limbs
            self._alias_id = id(self._data) if copy else data._alias_id
        else:
            if _xp is None:
                _xp = data.__array_namespace__()
            xp = self.xp = _xp
            if not xp.isdtype(data.dtype, 'integral'):
                raise TypeError(data.dtype)
            self._data = xp.astype(data[...,None], xp.uint64, copy=copy)
            self._limbs = 1
            self._alias_id = id(self._data) if copy else id(data)
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
        xp = x.xp
        limbs = max(x.limbs, y.limbs)
        alloc = limbs + 1
        x._alloc(alloc)

        y._alloc(alloc)
        if x._alias_id == y._alias_id:
            raise ValueError('edge case: detect overflow for in-place add to self. is a multiply reasonable here?')
            # this is simply to detect overflow!
            # nails might work better for this case
            y = NDBigInt(y, copy=True)
        x._data += y._data
        # in cases of overflow, the sum is less than the addend
        # if the end limb overflows then another is needed
        if xp.any(x._data[...,limbs-1] < y._data[...,limbs-1-1]):
            limbs = alloc
        # if a limb is all 0xf, as for negative numbers, there will be multiple chained overflows
        ref = y._data
        off = 0
        while True:
            oflows = x._data[...,off:-1] < ref[...,:-1]
            if not xp.any(oflows):
                break
            ref = xp.astype(oflows, xp.uint8, copy=False)
            off += 1
            x._data[...,off:] += ref
        x._limbs = limbs
        return x
    def __isub__(x, y):
        x._data ^= 0xffffffffffffffff
        x += y
        x._data ^= 0xffffffffffffffff
        return x
    def __eq__(x, y):
        xp = x.xp
        return xp.all(x._data == y._data, axis=-1)
    def _alloc(self, limbs):
        old_limbs = self._data.shape[-1]
        if old_limbs < limbs:
            xp = self.xp
            new_data = xp.empty([*self._data.shape[:-1], limbs], dtype=xp.uint64)
            new_data[...,:old_limbs] = self._data[...,:old_limbs]
            # sign extension
            #new_data[self._data[...,-1]>=UINT64_SIGN,old_limbs:] = UINT64_MAX
            new_data[...,old_limbs:] = (xp.astype(self._data[...,old_limbs-1], xp.int64, copy=False) >> 63)[...,None]
            self._data = new_data
        else:
            assert self.limbs <= limbs
            self._data = self._data[...,:limbs]
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

# notes on isub failure where the sign bit has overflowed into the first bit of the second limb
# ffffffffffffffff51565166835c8e83 - ffffffffffffffffc4d746b9eb97e454 = ffffffffffffffff8c7f0aac97c4aa2f
# ~0           5860961465801674371 - ~0          14183883315762160724 = ~0          10123822223749065263
#                                                                    before carry:
#                                                                       00000000000000008c7f0aac97c4aa2f
#                  carry check is different with subtraction vs addition
#                  in a subtraction, the lhs operand is normally larger than the rhs and the difference
#                  so comparison with the rhs operand is not necessarily easy or useful
#                  the operator that's easy to implement here is __irsub__ or __risub__ which doesn't quite exist
#                  but subtraction is also identical to addition with the negative
#                  as well as negation prior to addition
#                       the carry could maybe be calculated here via ... hrm
#                       well there could be a comparison with the rhs operand and the lhs first
#                       also this would be a borrow rather than a carry.
#                       the ideal check would be 8c > 51, diff > sum
#                       i suppose that's needed for positive numbers
#                       but one could also calculate the resulting sign bit.
#           we could compare with the negation of the subtractend.
#           diff < -subtractend == -diff > subtractend
#           this may mean copying the subtractend tho
#           when there could just be an in-place subtraction on the rhs
#           equivalent to in-place negation, then sum
#   another solution would be to negate the lhs, then add, then negate the result
# trying out ~x + y = ~z
# this consolidates the troubleshooting to the iadd function
# 0000000000000000aea9ae997ca3717c + ffffffffffffffffc4d746b9eb97e454 = 00000000000000007380f553683b55d0
#             12585782607907877244 +             -4262860757947390892 = 00000000000008322921849960486352
# 00000000000012585782607907877244 + ~0          14183883315762160724 = 00000000000008322921849960486352
#                                                            pre-carry: ~0           8322921849960486352
# so ideally the code would respond to 8<14 and add 1 to the ~0, producing the correct result
# this happens on the first iteration. it's allocated 3 limbs total.
# 0: ~0 ~0 8322921849960486352
# 1: ~0  0 8322921849960486352
# however it does not detect that 0 < ~0 to increment the next one.
# i think ref's indexing needs to be adjusted to compare correctly here.
# or the data's.
