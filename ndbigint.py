# representation:
#  2s complement, N uint64s as last dimension of ndarray
#  the 2s complement is considered over size N,
#    so the sign bit of the highest index value is the real sign bit
#  the reason to use the last dimension is to align with default broadcasting
#    and make initial implementation easier and clearer.
#  the first dimension might be more computationally efficient and would simply
#    mean permuting axes to broadcast consistently.

# next step: walk through a __mul__ between negative values and compare the calculation
# to normal integer multiplication

# normal integer multiplication of -3 and -4 with 2x32 bits each into 64 bit output
# 3=18446744073709551613
# 4=18446744073709551612
# lo3    =4294967293
# lo4    =4294967292
# hi3=hi4=4294967295
# lo3*lo4=18446744043644780556
# (hi3*hi4)<<64=0
# (hi3*lo4)<<32=17179869184
# (lo3*hi4)<<32=12884901888
# (18446744043644780556 + 0 + 17179869184 + 12884901888)%(1<<64) = 12

# the final sum in the code for -3 * -4 is reaching 7 instead of 12 due to the presence of further terms.
#   [
#       [18446744043644780556, 18446744056529682435],   # lo3 * lo4
#       [                   0, 18446744052234715140],
#       [         12884901888, 12884901888],            # (lo3*hi4) << 32
#       [                   0, 4294967296],
#       [         17179869184, 4294967296],             # (hi3*lo4) << 32
#       [                   0, 17179869184],
#       [                   0, 18446744065119617025],
#       [18446744065119617025, 0],
#       [                   0, 4294967292],
#       [          4294967292, 0],
#       [                   0, 4294967291],
#       [          4294967294, 0]
#   ]
# before restriding it looks like this:
#      [[[[18446744043644780556, 18446744056529682435, 0],  # lo3 * lo4
#         [18446744052234715140, 18446744065119617025, 0]],
#        [[         12884901888, 12884901888, 0],           # (lo3*hi4) << 32
#         [          4294967296, 4294967296, 0]],
#        [[         17179869184, 4294967296, 0],            # (hi3*lo4) << 32
#         [         17179869184, 4294967296, 0]]],
#       [[[                   0, 18446744065119617025, 18446744065119617025],
#         [                   0, 18446744065119617025, 18446744065119617025]],
#        [[                   0, 4294967292, 4294967292],
#         [                   0, 4294967294, 4294967294]],
#        [[                   0, 4294967291, 4294967294],
#         [                   0, 4294967291, 4294967294]]]]

# the problem appears to be that the second set of matrices, the ones
# offset by 1, are missing trailing zeros to wrap during restriding.
# (note the "oops no-op" comment showing failure to fully adjust the final
#  bounds correctly when merging the concepts of limb count, analogous to
#  an internal failure to remember and include concepts of setting them)

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
                self._data = xp.asarray(data, copy=copy)
                self._limbs = _limbs
            elif xp.isdtype(data.dtype, xp.uint64) and xp.any(data >= 0x8000000000000000):
                copy = True
                if alloc is None or alloc < 2:
                    alloc = 2
                self._data = xp.empty([*data.shape, alloc], dtype=xp.uint64)
                self._data[...,0] = data
                self._data[...,1] = 0
                self._limbs = 2
                alloc = None
            else:
                self._data = xp.astype(data[...,None], xp.uint64, copy=bool(copy))
                self._limbs = 1
        self._view = copy is False
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
    def sum(x, *, axis = None, keepdims = False, _trunc = None):
        if _trunc is None:
            _trunc = x._limbs + 1
            x._alloc(x._limbs + 1)
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
                x = NDBigInt(y[*preceding_axes,:midsize+1,...]._data, copy=copy, _xp=xp, _limbs=y._limbs)
                x_sum = x[*preceding_axes,:midsize,...]
                x_sum = x_sum.__iadd__(y[*preceding_axes,midsize+1:,...], _trunc=_trunc)
            else:
                x = NDBigInt(y[*preceding_axes,:midsize,...]._data, copy=copy, _xp=xp, _limbs=y._limbs)
                x = x.__iadd__(y[*preceding_axes,midsize:,...], _trunc=_trunc)
            copy = False
            size = x._data.shape[axis]

        if not keepdims:
            x = x[*preceding_axes,0,...]
        x._view = None
        return x

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

    def __iadd__(x, y, _trunc=None):
        xp = x.xp
        #x_list = x._tolist()
        #y_list = y._tolist()
        #expected_sum = x._tolist(y, visitor=lambda x, y: int(x) + int(y))
        if _trunc is None:
            limbs = max(x.limbs, y.limbs)
            _trunc = limbs + 1
        else:
            limbs = _trunc
        alloc = _trunc
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
        # one way to reduce the iterations here could be to persist overflow data
        ref = y._data[...,:alloc]
        off = 0
        while True:
            oflows = x._data[...,off:alloc-1] < ref[...,:-1]
            if not xp.any(oflows):
                break
            ref = xp.astype(oflows, xp.uint8, copy=False)
            off += 1
            x._data[...,off:alloc] += ref


        # set limbs
        # probably efficiency improvements exist
        if alloc > limbs and xp.any((x._data[...,limbs] ^ x._data[...,limbs-1]) >= 0x80000000_00000000):
            # (a&63rd)==(b&63rd) is ~(a&63rd)^(b&63rd) or (a^b)&63rd == 0 given xor is bitwise

            # the extra limb's sign bit is needed
            # this branch could likely be simplified away by starting with limbs = limbs + 1 and reducing it
            limbs += 1
        else:
            # this doesn't need to allocate an entire copy of the data here
            signed_x = xp.astype(x._data, xp.int64)
            while limbs > 1 and xp.all(signed_x[...,limbs-1] == signed_x[...,limbs-2]>>63):
                limbs -= 1
        x._limbs = limbs
        #actual_sum = x._tolist(); assert expected_sum == actual_sum
        return x
    def __mul__(x, y):
        xp = x.xp
        #if _may_share_memory(xp, x._data, y._data):
        #    raise NotImplementedError('in place overlapping multiply')

        if x._data[-1] & 0x8000000000000000 or y._data[-1] & 0x8000000000000000:
            import pdb; pdb.set_trace()
            #raise NotImplementedError('product of negative')

        # This approach uses masking and shifting which could be reduced if the
        # data were directly cast from uint64 to uint32 without loss of
        # elements. A function like _may_share_memory could be added to do this
        # if one doesn't exist, with fallback to the below code.

        # The matmuls present in this could likely be unified somehow.

        limbs = x._limbs + y._limbs # assuming negative numbers present in both operands
        x._alloc(limbs)
        y._alloc(limbs)
        x = x._data
        y = y._data
        shape = x.shape[:-1]

        # halflimb product that would sum along axis -2:
        # sum([
        #   [xlo[0]*ylo[0], xlo[0]*yhi[0], xlo[0]*ylo[1], xlo[0]*yhi[1],             0,             0,             0],
        #   [            0, xhi[0]*ylo[0], xhi[0]*yhi[0], xhi[0]*ylo[1], xhi[0]*yhi[1],             0,             0],
        #   [            0,             0, xlo[1]*ylo[0], xlo[1]*yhi[0], xlo[1]*ylo[1], xlo[1]*yhi[1],             0],
        #   [            0,             0,             0, xhi[1]*ylo[0], xhi[1]*yhi[0], xhi[1]*ylo[1], xhi[1]*yhi[1]],
        # ], axis=-2)
        # is this outer product with stride reduced by one:
        #   [xlo[0]*ylo[0], xlo[0]*yhi[0], xlo[0]*ylo[1], xlo[0]*yhi[1],             0,             0,             0,             0],
        #   [xhi[0]*ylo[0], xhi[0]*yhi[0], xhi[0]*ylo[1], xhi[0]*yhi[1],             0,             0,             0,             0],
        #   [xlo[1]*ylo[0], xlo[1]*yhi[0], xlo[1]*ylo[1], xlo[1]*yhi[1],             0,             0,             0,             0],
        #   [xhi[1]*ylo[0], xhi[1]*yhi[0], xhi[1]*ylo[1], xhi[1]*yhi[1],        unused,        unused,        unused,        unused],

        #  whole limb products that would sum along axis -2:
        #   [[  xl[0]*yl[0]      ,   xl[0]*yl[1]      ,                   0,                   0],
        #    [                  0,   xl[1]*yl[0]      ,   xl[1]*yl[1]      ,                   0]],
        #   [[((xl[0]*yh[0])<<32), ((xl[0]*yh[1])<<32),                   0,                   0],
        #    [                  0, ((xl[1]*yh[0])<<32), ((xl[1]*yh[1])<<32),                   0]],
        #   [[((xh[0]*yl[0])<<32), ((xh[0]*yl[1])<<32),                   0,                   0],
        #    [                  0, ((xh[1]*yl[0])<<32), ((xh[1]*yl[1])<<32),                   0]],
        #   [[                  0,   xh[0]*yh[0]      ,   xh[0]*yh[1]      ,                   0],
        #    [                  0,                   0,   xh[1]*yh[0]      ,   xh[1]*yh[1]      ]],
        #   [[                  0, ((xl[0]*yh[0])>>32), ((xl[0]*yh[1])>>32),                   0],
        #    [                  0,                   0, ((xl[1]*yh[0])>>32), ((xl[1]*yh[1])>>32)]],
        #   [[                  0, ((xh[0]*yl[0])>>32), ((xh[0]*yl[1])>>32),                   0],
        #    [                  0,                   0, ((xh[1]*yl[0])>>32), ((xh[1]*yl[1])>>32)]],
        # come from these outer products:
        #   [[  xl[0]*yl[0]      ,   xl[0]*yl[1]      ,                   0,                   0,                   0],
        #    [  xl[1]*yl[0]      ,   xl[1]*yl[1]      ,                   0,              unused,              unused]],
        #   [[((xl[0]*yh[0])<<32), ((xl[0]*yh[1])<<32),                   0,                   0,                   0],
        #    [((xl[1]*yh[0])<<32), ((xl[1]*yh[1])<<32),                   0,              unused,              unused]],
        #   [[((xh[0]*yl[0])<<32), ((xh[0]*yl[1])<<32),                   0,                   0,                   0],
        #    [((xh[1]*yl[0])<<32), ((xh[1]*yl[1])<<32),                   0,              unused,              unused]],
        #   [[                  0,   xh[0]*yh[0]      ,   xh[0]*yh[1]      ,                   0,                   0],
        #    [                  0,   xh[1]*yh[0]      ,   xh[1]*yh[1]      ,              unused,              unused]],
        #   [[                  0, ((xl[0]*yh[0])>>32), ((xl[0]*yh[1])>>32),                   0,                   0],
        #    [                  0, ((xl[1]*yh[0])>>32), ((xl[1]*yh[1])>>32),              unused,              unused]],
        #   [[                  0, ((xh[0]*yl[0])>>32), ((xh[0]*yl[1])>>32),                   0,                   0],
        #    [                  0, ((xh[1]*yl[0])>>32), ((xh[1]*yl[1])>>32),              unused,              unused]],

        # so the whole product-sum parts can be expressed as a restrided ndarray with an extra dimension 6 large
        # or two extra dimensions of shape (2,3)
        # where the 6 components are:
        # 1/1,1 low halflimb products
        # 2/1,2 high halflimb of low*high halflimb products
        # 3/1,3 high halflimb of high*low halflimb products
        # 4/2.1 high halflimb products (offset by 1 limb)
        # 5/2.2 low halflimb of low*high halflimb products (offset by 1 limb)
        # 6/2.3 low halflimb of high*low halflimb products (offset by 1 limb)

        prod = xp.empty(
            [*shape, 2, 3, limbs, limbs + 1], # this + 1 is to allow for restriding to offset the values prior to taking their sum.
            dtype = xp.uint64
        )


        # something i might want to figure out for signed multiplication of 64 bit values is how to do signed multiplication of 2-bit values.
        # -3 = 11 01
        # -2 = 11 10
        # -3 * -2 = 6 = 00 01 10
        # 
        #          11 11 01
        #        x 11 11 10
        #     -------------
        #    11 11
        #  1 11 11 1
        # 11 11 11 11       # carry
        #                
        #          _0 _0 _0 # _1 _1 _1 x .. .. _0
        #          1_ 1_ 1_ # _1 _1 _1 x .. .. 1_
        #          0_ 0_ 0_ # 1_ 1_ 0_ x .. .. _0
        #        1 _1 _0 _  # 1_ 1_ 0_ x .. .. 1_

        #       _1 _1 _1    # _1 _1 _1 x .. _1 ..
        #       1_ 1_ 1_    # _1 _1 _1 x .. 1_ ..
        #       1_ 1_ 0_    # 1_ 1_ 0_ x .. _1 ..
        #     1 _1 _0 _     # 1_ 1_ 0_ x .. 1_ ..

        #    _1 _1 _1       # _1 _1 _1 x _1 .. ..
        #    1_ 1_ 1_       # _1 _1 _1 x 1_ .. ..
        #    1_ 1_ 0_       # 1_ 1_ 0_ x _1 .. ..
        #  1 _1 _0 _        # 1_ 1_ 0_ x 1_ .. ..
        # ------------------
        # 11 00 11 00 01 10

        # 
        #             11 01
        #        x    11 10
        #     -------------
        #           
        #       11 11       # carry
        #                
        #             _0 _0 #    _1 _1 x    .. _0
        #             1_ 1_ #    _1 _1 x    .. 1_
        #             0_ 0_ #    1_ 0_ x    .. _0
        #           1 _0 _  #    1_ 0_ x    .. 1_

        #          _1 _1    #    _1 _1 x    _1 ..
        #          1_ 1_    #    _1 _1 x    1_ ..
        #          1_ 0_    #    1_ 0_ x    _1 ..
        #        1 _0 _     #    1_ 0_ x    1_ ..
        # ------------------
        #       10 11 01 10

        # 3 2bits * 3 2bits -> 3 2bits
        # 2 2bits * 2 2bits -> 2 2bits
        # it looks negative values need to be sign extended to as far as the final product


        # construct the outer products of halflimbs masked and shifted to
        # collect overflow and carry information and padded with zeros

        #x_lo = xp.astype(x[...,:xlimbs], xp.uint32, copy=False)
        #y_lo = xp.astype(y[...,:ylimbs], xp.uint32, copy=False)
        x_lo = x[...,:limbs] & 0x00000000ffffffff
        y_lo = y[...,:limbs] & 0x00000000ffffffff
        x_hi = x[...,:limbs] >> 32
        y_hi = y[...,:limbs] >> 32
        # low halflimb products are in-place.
        #if NDBigInt((xp.reshape(x[...,:limbs],-1)[0], xp.int64) < 0 and:
        #    import pdb; pdb.set_trace()
        prod[..., 0, 0, :, :limbs] = x_lo[...,None] @ y_lo[...,None,:]
        # low*high halflimb products are shifted up by a halflimb
        prod[..., 0, 1, :, :limbs] = x_lo[...,None] @ y_hi[...,None,:]
        prod[..., 0, 2, :, :limbs] = x_hi[...,None] @ y_lo[...,None,:]
        prod[..., 1, 1:, :, 1:limbs+1] = prod[..., 0, 1:, :, :limbs]
        prod[..., 0, 1:, :, :] <<= 32
        prod[..., 1, 1:, :, 1:limbs+1] >>= 32
        # high halflimb products are shifted up a whole limb
        prod[..., 1, 0, :, 1:limbs+1] = x_hi[...,None] @ y_hi[...,None,:]
        # zeros in the unused limbs
        prod[..., 1, :, :, 0] = 0
        prod[..., 0, :, :, limbs:] = 0
        prod[..., 1, :, :, limbs+1:] = 0 # oops no-op ?

        # reshape with the padded dimension 1 size smaller (limbs)
        # to give the outer product the slided offsetting for the sum
        prod = xp.reshape(
            xp.reshape(
                prod,
                [*shape, 6, -1]
            )[..., :limbs * limbs],
            [*shape, 6 * limbs, limbs]
        )

        # then the product might be a bigint sum of prod along -2

        prod = NDBigInt(prod, _xp=xp, _limbs=limbs)
        return prod.sum(axis=-2, _trunc=limbs)

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
    def __int__(self):
        xp = self.xp
        signlimb = self.limbs - 1
        accum = int(xp.astype(self._data[...,signlimb], xp.int64, copy=False))
        for item in xp.unstack(self._data[...,:signlimb][...,::-1]):
            accum <<= 64
            accum += int(item)
        return accum
    def __getitem__(self, slices):
        return NDBigInt(self._data[*slices, :], copy=False, _xp=self.xp, _limbs=self._limbs)
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
    def _alloc(self, alloc, _sign_extend=True):
        old_alloc = self._data.shape[-1]
        if old_alloc < alloc:
            assert not self._view and "resizing through view might indicate implementation of view functionality in ndarray.py and using it for resizing here"
            new_data = self.xp.empty([*self._data.shape[:-1], alloc], dtype=xp.uint64)
            new_data[...,:old_alloc] = self._data[...,:old_alloc]
            self._data = new_data
        else:
            assert self._limbs <= alloc
            #self._data = self._data[...,:alloc]
        if alloc > self._limbs and _sign_extend:
            self._sign_extend(self.xp, self._data, old_alloc)
            #self._data[...,old_alloc:] = self.xp.astype(self._data[...,old_alloc-1,None], xp.int64, copy=False) >> 63
    @staticmethod
    def _sign_extend(xp, data, start_limb):
        #new_data[self._data[...,-1]>=UINT64_SIGN,old_alloc:] = UINT64_MAX
        #self._data[...,old_alloc:] = self.xp.astype(self._data[...,old_alloc-1,None], xp.int64, copy=False) >> 63
        #signed_data = xp.astype(data, xp.int64, copy=False) # always makes copy when dtypes differ

        signs = xp.astype(data[..., start_limb-1, None], xp.int64) # reduce new allocation size to 1 limb
        signs >>= 63
        data[..., start_limb:] = signs

        # haven't found an approach yet to do it without allocating new data
        #block = data[..., start_limb:]
        #assert _may_share_memory(block, data)
        #block[:] = data[..., start_limb-1,None]
        #block &= 0x8000000000000000
        #block[block] = 0xffffffffffffffff # here i think i incorrectly use an int as a bool

if __name__ == '__main__':
    import array_api_strict as xp

    assert int(NDBigInt(xp.asarray(-6532100632237123854),alloc=3) + NDBigInt(xp.asarray(7958265450555812818),alloc=2)) == 1426164818318688964
    assert int(NDBigInt(xp.asarray(7692698082559361259)) + NDBigInt(xp.asarray(7692698082559361259))) == 15385396165118722518
    assert int(NDBigInt(xp.asarray(15761168082059424201)) + NDBigInt(xp.asarray(8937115293130262283))) == 24698283375189686484

    assert int(NDBigInt(xp.asarray(3)) * NDBigInt(xp.asarray(4))) == 12
    assert int(NDBigInt(xp.asarray(7692698082559361259)) * NDBigInt(xp.asarray(7692698082559361259))) == 59177603789412473292821695494070065081
    assert int(NDBigInt(xp.asarray(15761168082059424201)) * NDBigInt(xp.asarray(8937115293130262283))) == 140859376303769844698647197831087710883
    assert int(NDBigInt(xp.asarray(7692698082559361259)) * NDBigInt(xp.asarray(7692698082559361259)) + NDBigInt(xp.asarray(15761168082059424201)) * NDBigInt(xp.asarray(8937115293130262283))) == 200036980093182317991468893325157775964
    assert int(NDBigInt(xp.asarray(7692698082559361259)) * NDBigInt(xp.asarray(7692698082559361259)) * NDBigInt(xp.asarray(15761168082059424201)) * NDBigInt(xp.asarray(8937115293130262283))) == 8335720360928247907391432232202715839389412648863354718493794045983121976523
    # it might make sense to review potential simplification of sign extension in __iadd__ before implementing multiplication of negative
    # numbers, so as to consider whether what is learned is helpful when representing negative products.
    assert int(NDBigInt(xp.asarray(-3)) * NDBigInt(xp.asarray(-4))) == 12
    # 0xfffc # -3
    # 0xfffd # -4
    # ---------
    #...000c     
    #assert int(NDBigInt(xp.asarray(-3)) * NDBigInt(xp.asarray(4))) == -12
    #assert int(NDBigInt(xp.asarray(3)) * NDBigInt(xp.asarray(-4))) == -12
    #assert int(NDBigInt(xp.asarray(-6532100632237123854),alloc=3) * NDBigInt(xp.asarray(7958265450555812818),alloc=2)) == -51984190781086484234522341753506760572

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
