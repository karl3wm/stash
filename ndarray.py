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
        xp = self.xp = data.__array_namespace__()
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
        return 'NDArray('+repr(self.data).replace('\n','\n        ')+')'
    def __str__(self):
        return str(self.data)

    def _reserve(self, shape):
        xp = self.xp
        capacity = xp.max(xp.stack([self.capacity, shape]), axis=0)
        if xp.any(capacity != self.capacity):
            capacity = ceil_exp_2(capacity)
            storage = xp.empty(capacity, dtype=self.storage.dtype)
            return [storage, capacity]
        else:
            return [self.storage, self.capacity]

    def resize(self, shape):
        xp = self.xp
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

    def insert_empty(self, where, expansion):
        xp = self.xp
        # resizes the ndlist to prepare for insertion of data, leaving empty regions.
        # the unassigned regions form an n-dimensional "+" shape extending in
        # every axis, with the center volume of the "+" located at 'where' with
        # size 'expansion'.
        # returns a list of nonoverlapping views over the newly-created unallocated volumes, to place data into.
        # each view has attributes .data and .bounds and supports indexing

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
        axes_expanding_idcs = xp.nonzero(axes_expanding_mask)[0]
        nexpanding = axes_expanding_idcs.shape[0]
        move_region_idcs = xp.reshape(
            xp.stack(xp.meshgrid(*([xp.arange(2)]*nexpanding),indexing='ij')), # np.indices
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
            bounds_empty = xp.empty([2, self.ndim], dtype=self.dtype_index)
            bounds_empty[0,:] = 0
            bounds_empty[1,:] = new_shape
            ray_dimensions = False

            # elements of move_region_idx are 1 for axes needing expansion
            for idx in range(nexpanding):
                axis = int(axes_expanding_idcs[idx])
                if move_region_idx[idx]:
                    # region axis shifts up from insertion
                    slicelist_move_src[axis] = slice(lower[axis], old_shape[axis])
                    upper_slice = slice(upper[axis], new_shape[axis])
                    slicelist_move_dst[axis] = upper_slice
                    if ray_dimensions:
                        slicelist_empty[axis] = upper_slice
                        bounds_empty[0,axis] = upper[axis]
                    else:
                        slicelist_empty[axis] = slice(lower[axis], upper[axis])
                        bounds_empty[0,axis] = lower[axis]
                        bounds_empty[1,axis] = upper[axis]
                        ray_dimensions = True
                else:
                    # region axis is below insertion
                    lower_slice = slice(None, lower[axis])
                    slicelist_move_src[axis] = slicelist_move_dst[axis] = lower_slice
                    if ray_dimensions:
                        slicelist_empty[axis] = lower_slice
                        bounds_empty[1,axis] = lower[axis]
            empty_views.append([bounds_empty, slicelist_empty])
            storage[*slicelist_move_dst] = self.storage[*slicelist_move_src]
        self.storage = storage
        self.capacity = capacity
        self.data = storage[*slicelist_dst_data]
        self.shape = new_shape
        return [self.Subview(self, bounds, *slices) for bounds, slices in empty_views]
        
    def insert(self, axis, offset, data):
        xp = self.xp
        # expands along only one dimension
        where, expansion = xp.unstack(xp.zeros([2,self.ndim],dtype=self.dtype_index))
        where[axis] = offset
        expansion[axis] = data.shape[axis]
        insertion_shape = xp.asarray(self.shape, copy=True)
        insertion_shape[axis] = data.shape[axis]
        assert xp.all(xp.asarray(data.shape) == insertion_shape)
        hole, = self.insert_empty(where, expansion)
        assert xp.all(hole.bounds[0,:] == where)
        assert hole.bounds[1,axis] - hole.bounds[0,axis] == expansion[axis]
        assert xp.all(hole.bounds[1,:axis] - hole.bounds[0,:axis] == self.shape[:axis])
        assert xp.all(hole.bounds[1,axis+1:] - hole.bounds[0,axis+1:] == self.shape[axis+1:])
        hole.data[:,...] = data

    class Subview:
        def __init__(self, ndarray, bounds, *slices):
            self.parent = ndarray
            self.data = ndarray.data[slices]
            self.bounds = bounds
        def __getitem__(self, slices):
            return self.data[slices]
        def __setitem__(self, slices, vals):
            self.data[slices] = vals
        def __getattr__(self, attr):
            return getattr(self.data, attr)
        def __repr__(self):
            return ( 'NDArray.Subview(' +
                (repr(self.parent)+',\nbounds=').replace('\n','\n                ') +
                repr(self.bounds).replace('\n','\n                       ') +
            ')')
        def __str__(self):
            return str(self.data)

if __name__ == '__main__':
    import array_api_strict as xp

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

    ndlist3 = NDArray(xp.asarray([[[1,2],[3,4]],[[5,6],[7,8]]]))
    gaps = ndlist3.insert_empty([1,1,1],[1,1,1])
    for hole in gaps:
        hole[:,...] = sum(xp.meshgrid(xp.arange(3),xp.arange(3),xp.arange(3)))[tuple(slice(bound[0],bound[1]) for bound in xp.unstack(hole.bounds.T))]
    assert xp.all(ndlist3.data == xp.asarray([[[1,1,2],[1,2,3],[3,3,4]],[[1,2,3],[2,3,4],[3,4,5]],[[5,3,6],[3,4,5],[7,5,8]]]))
