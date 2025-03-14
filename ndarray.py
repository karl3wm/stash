import array_api_strict as xp

def ceil_exp_2(data):
    data = xp.asarray(data)
    if len(data.shape) == 0:
        return 1 << (int(data) - 1).bit_length()
    else:
        return xp.reshape(
                xp.asarray([
                    (1 << (int(i) - 1).bit_length())
                    for i in xp.reshape(data,-1)
                ], dtype=data.dtype),
            data.shape)

class NDArray:
    def __init__(self, data):
        self._xp = data.__array_namespace__()
        assert self._xp is xp
        self.ndim = len(data.shape)
        self.dtype_index = xp.__array_namespace_info__().default_dtypes(device=data.device)['indexing']
        self.capacity, self.shape = xp.unstack(xp.zeros([2, self.ndim], dtype=self.dtype_index))
        self.storage = xp.empty(self.shape, dtype=data.dtype)
        self.data = self.storage
        self.resize(data.shape)
        self.data[:,...] = data
    def __getitem__(self, idcs):
        return self.data[idcs]
    def __setitem__(self, idcs, data):
        self.data[idcs] = data
    def __repr__(self):
        return 'NDArray('+repr(self.data)+')'
    def __str__(self):
        return str(self.data)
    def _reserve(self, shape):
        capacity = xp.max(xp.stack([self.capacity, shape]), axis=0)
        if xp.any(capacity != self.capacity):
            capacity = ceil_exp_2(capacity)
            storage = xp.empty(capacity, dtype=self.storage.dtype)
            return [storage, capacity]
        else:
            return [self.storage, self.capacity]
    def resize(self, shape):
        shape = xp.asarray(shape)
        storage, capacity = self._reserve(shape)
        if storage is not self.storage:
            shared_shape = xp.min(xp.stack([shape, self.shape]), axis=0)
            shared_slice = tuple([slice(0,x) for x in shared_shape])
            storage[shared_slice] = self.storage[shared_slice]
            self.storage = storage
            self.capacity = capacity
        self.shape = shape
        self.data = storage[*[slice(0,x) for x in shape]]
    def _insert_empty(self, where, expansion):
        # resizes the ndlist to prepare for insertion of data,
        # leaving unallocated regions at 'where' of size 'expansion'
        # the unallocated regions form an n-dimensional "+" shape extending in every axis

        # returns a new list of slices over the entire final shape for convenience
        #  let's change this to return a list of views over the newly created areas

        # note: this could support ragged nd data with an additional parameter specifying which axes to shift old data

        lower = xp.asarray(where)
        expansion = xp.asarray(expansion)
        upper = lower + expansion
        old_shape = self.shape
        new_shape = xp.asarray(old_shape, copy=True)
        new_shape[:expansion.shape[0]] += expansion
        storage, capacity = self._reserve(new_shape)
        empty_views = []

        axes_expanding_mask = (expansion != 0)
        #axes_expanding_idcs = xp.argwhere(axes_expanding_mask)[:,0]
        axes_expanding_idcs = xp.nonzero(axes_expanding_mask)[0]
        nexpanding = axes_expanding_idcs.shape[0]
        #move_region_idcs = xp.reshape(xp.indices(xp.full(nexpanding,2)), nexpanding,-1).T
        move_region_idcs = xp.reshape(
            xp.stack(xp.meshgrid(*([xp.arange(2)]*nexpanding))), # np.indices
            [nexpanding, -1]
        ).T
        slicelist_src_data = [ slice(None, old_shape[idx]) for idx in range(self.ndim) ]
        slicelist_dst_data = [ slice(None, new_shape[idx]) for idx in range(self.ndim) ]

        # copy unmoved data if storage was reallocated
        if storage is not self.storage:
            slicelist_move = list(slicelist_src_data)
            for axis in axes_expanding_idcs:
                slicelist_move[axis] = slice(None, lower[axis])
            slicelist_move = tuple(slicelist_move)
            storage[slicelist_move] = self.storage[slicelist_move]

        # slide the moved data
        for move_region_idx in xp.unstack(move_region_idcs[1:,...]):
            slicelist_move_src = list(slicelist_src_data)
            slicelist_move_dst = list(slicelist_dst_data)
            slicelist_empty = list(slicelist_dst_data)
            # elements of move_region_idx are 1 for axes needing expansion
            for idx in range(nexpanding):
                axis = axes_expanding_idcs[idx]
                if move_region_idx[idx]:
                    # region axis shifts up from insertion
                    slicelist_move_src[axis] = slice(lower[axis], old_shape[axis])
                    slicelist_move_dst[axis] = slice(upper[axis], new_shape[axis])
                    slicelist_empty[axis] = slice(lower[axis], upper[axis])
                else:
                    # region axis is below insertion
                    slicelist_move_src[axis] = slicelist_move_dst[axis] = slice(None, lower[axis])
            empty_views.append(storage[*slicelist_empty])
            storage[*slicelist_move_dst] = self.storage[*slicelist_move_src]
        self.storage = storage
        self.capacity = capacity
        self.data = storage[*slicelist_dst_data]
        self.shape = new_shape
        #return slicelist_dst_data
        return empty_views
        
    def insert(self, axis, offset, data):
        # expands along only one dimension
        where, expansion = xp.unstack(xp.zeros([2,self.ndim],dtype=self.dtype_index))
        where[axis] = offset
        expansion[axis] = data.shape[axis]
        insertion_shape = xp.asarray(self.shape, copy=True)
        insertion_shape[axis] = data.shape[axis]
        assert xp.all(xp.asarray(data.shape) == insertion_shape)
        #slicelist = self._insert_empty(where, expansion)
        #slicelist[axis] = slice(offset, offset + expansion[axis])
        #self.data[*slicelist] = data
        hole, = self._insert_empty(where, expansion)
        hole[:,...] = data

if __name__ == '__main__':
    ndlist1 = NDArray(xp.asarray([1,2,3]))
    assert xp.all(ndlist1.data == xp.asarray([1,2,3]))
    ndlist1.insert(0, 2, xp.asarray([4,5]))
    assert xp.all(ndlist1.data == xp.asarray([1,2,4,5,3]))

    ndlist2 = NDArray(xp.asarray([[1,2],[3,4]]))
    assert xp.all(ndlist2.data == xp.asarray([[1,2],[3,4]]))
    ndlist2.insert(0, 1, xp.asarray([[5,6],[7,8]]))
    assert xp.all(ndlist2.data == xp.asarray([[1,2],[5,6],[7,8],[3,4]]))
    ndlist2.insert(1, 1, xp.asarray([[9,10],[11,12],[13,14],[15,16]]))
    assert xp.all(ndlist2.data == xp.asarray([[1,9,10,2],[5,11,12,6],[7,13,14,8],[3,15,16,4]]))
