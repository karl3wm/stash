
class NDBigInt:
    def __init__(self, data):
        if type(data) is NDBigInt:
            self._data = data._data
            self.xp = data.xp
        else:
            xp = self.xp = data.__array_namespace__()
            # the initial data must already be an ndarray of integers
            if not xp.isdtype(data.dtype, 'integral'):
                raise TypeError(data.dtype)
            self._data = xp.astype(data[...,None], xp.uint64)
            self._expandshift()
    def _reserve(self, new_words):
        old_words = self._data.shape[-1]
        if old_words < new_words:
            new_data = xp.empty([*self._data.shape[:-1], new_words], dtype=xp.uint64)
            new_data[...,:old_words] = self._data
            new_data[...,old_words:] = 0
            self._data = new_data
    def _expandshift(self):
        # each highest bit is moved into the lowest bit of adjacent val
        # maybe i'll just repeat that, expanding once when needed
        xp = self.xp
        min_word = 0
        end_word = self._data.shape[-1]
        hi_bit = 1<<63
        while True:
            hi_bit_mask = xp.bitwise_and(self._data[...,min_word:], hi_bit)
            if not xp.any(hi_bit_mask):
                break
            if xp.any(hi_bit_mask[...,-1]):
                # highest bit set, grow _data
                end_word += 1
                self._reserve(end_word)
                self._data[...,end_word] = hi_bit_mask[...,-1] >> 63
                working_data = self._data[...,min_word:-1]
            else:
                working_data = self._data[...,min_word:end_word]
            working_data[...] ^= hi_bit_mask
            min_word += 1
            working_data[...,1:] += hi_bit_mask[...,:-1] >> 63
    def __int__(self):
        accum = 0
        for item in self.xp.unstack(self._data[...,::-1]):
            accum <<= 64
            accum += int(item)
        return accum
    def __getitem__(self, slices):
        item = NDBigInt(self)
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
        
#def __NDBigIntOpWordwiseUnaryAllocating(opname):
#    def op(a, *params, **kwparams):
#        c = NDBigInt(getattr(a._data, opname)(*params, **kwparams))
#        c._expandshift()
#        return c
#    op.__name__ = opname
#    return op
def __NDBigIntOpWordwiseBinaryInplace(opname):
    def op(a, b, *params, **kwparams):
        if type(b) is not NDBigInt:
            b = NDBigInt(b)
        b_depth = b._data.shape[-1]
        a._reserve(b_depth+1)
        getattr(a._data[...,:b_depth], opname)(b._data, *params, **kwparams)
        a._expandshift()
        return a
    op.__name__ = opname
    return op
for opname, factory in [
        [f'__{opname}__', factory]
        for opname, factory
        in [
            [prefix + opname, factory]
            for prefix, factory in [
                #['',__NDBigIntOpBinaryAllocating],
                #['r',__NDBigIntOpBinaryAllocating],
                ['i',__NDBigIntOpWordwiseBinaryInplace]
            ]
            for opname in [
                'add','sub'#,'mul','matmul','floordiv',
                #'mod','divmod',
                #'lshift',
                'and','xor','or'
            ]
        #] + [
        #    ['getitem', __NDBigIntOpWordwiseUnaryAllocating]
        ]
]:
    setattr(NDBigInt, opname, factory(opname))

if __name__ == '__main__':
    import array_api_strict as xp
    ar = NDBigInt(xp.asarray([1,2,3]))
    ar += xp.asarray([1,2,3])
    print(ar)
