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
        # resizes the ndlist to prepare for insertion of data, leaving empty regions.
        # the unassigned regions form an n-dimensional "+" shape extending in
        # every axis, with the center volume of the "+" located at 'where' with
        # size 'expansion'.
        # returns a list of views over the newly-created unallocated volumes, to place data into.
        # the views are returned in axis order with the center of the "+" returned last.

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
            bounds_empty = xp.empty([2, self.ndim], dtype=self.dtype_index)
            bounds_empty[0,:] = 0
            bounds_empty[1,:] = new_shape
            ray_dimensions = False

            # if an element of move_region_idx is 1 (or 0)
            # then all following (or preceding) elements of the empty view wou

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
        # expands along only one dimension
        where, expansion = xp.unstack(xp.zeros([2,self.ndim],dtype=self.dtype_index))
        where[axis] = offset
        expansion[axis] = data.shape[axis]
        insertion_shape = xp.asarray(self.shape, copy=True)
        insertion_shape[axis] = data.shape[axis]
        assert xp.all(xp.asarray(data.shape) == insertion_shape)
        hole, = self._insert_empty(where, expansion)
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

#
# 0 1  4 5
# 2 3  6 7
#
# 0 - 1  - - -  4 - 5
# - - -  - - -  - - -
# 2 - 3  - - -  6 - 7
#
# 7 moved volumes
# 3 inserted planes
# 8 inserted volumes between moves
# 1 central inserted volume
# 6 inserted volumes neither central nor between moves

# it looks like the current implemented would first return the inserted planes, 
# then the bars connecting them,
# then the central volume
#  it does this but not in that order. total: 3 2-long,1-short volumes, 3 1-long,2-short volumes, 1 central 3-short volume.
#  we can actually count the dimensions and return the data however desired. i think the interest is in separate nonoverlapping minimally-sized volumes.
# 
#       the planes and bars might be more intuitive ...
#           mostly we need the coordinates of the areas; a way to algorithmically fill them correctly
#       so maybe it's helpful to have them nonoverlapping for correctly filling them with data
# 
# enumerate nonoverlapping volumes
# say we engage '1' first. there is one nonverlapping volume between it and 0.
# if we then move to '2' there is a clear nonoverlapping volume between it and 0.
#   however in moving to '2' we are also increasing a new dimension.
#   when increasing this new dimension we cross a nonoverlapping volume of size 3x1 between 2-3 and 0-1.
# then '3' covers a nonoverlapping volume between '2' and '3'.
#
#   each dimension has a different span across values. when we progress dimension 0 we cross small overlapping volumes.
#   when we progress dimension 1 we cross 3x1 nonverlapping valumes (and dimension 2 is 3x3x1 volumes) -- but these happen less frequently.

# in that iteration, there are 4 1x1 1d volumes, 2 3x1 2d volumes, and 1 3x3x1 3d volume.
# that is actually a total of 7 volumes. they could be each equated with one of the iterated moved areas.
#
# 0 - 1  - - -  4 - 5
# - - -  - - -  - - -
# 2 - 3  - - -  6 - 7
#
#   1: 1x1 0-1
#   2? 3x1 ---
#   3? 1x1 2-3
#   4? 3x3x1 ---/---/---
#   5? 1x1 4-5
#   6? 3x1 ---
#   7? 1x1 6-7
# or
#   0: 1x1 0-1
#   1? 3x1 ---
#   2? 1x1 2-3
#   3? 3x3x1 ---/---/---
#   4? 1x1 4-5
#   5? 3x1 ---
#   6? 1x1 6-7
# the order may be the same whether one considers conditioning on a 0 coord or a 1 coord; i may have made a mistake
# so it may make sense to do the 1-based iteration
#   1: [1,0,0] : 1x1x1 0-1      deepdims=0  shortdims=3
#   2: [0,1,0] : 3x1x1 01-23    deepdims=1  shortdims=2
#   3: [1,1,0] : 1x1x1 2-3      deepdims=0  shortdims=3
#   4: [0,0,1] : 3x3x10123-4567 deepdims=2  shortdims=1
#   5: [1,0,1] : 1x1x1 4-5      deepdims=0  shortdims=3
#   6: [1,1,0] : 3x1x1 45-67    deepdims=1  shortdims=2
#   7: [1,1,1] : 1x1x1 6-7      deepdims=0  shortdims=3
#   ...[0,0,0,1]..............  deepdims=3  shortdims=0  if 4d
# i'm trying to figure out the pattern of deepdims/shortdims.
# it's a clear pattern conditioned on the index number .. uh ... kinda like (idx%2 == 0) + (idx%4 == 0) + (idx%8 == 0)
# if the indices were sparse instead of dense, it might be clearer
#  these are a flattened list of coordinates, but the original list was created by combining a bunch of [1,0] pairs
#  in that original list, the pattern is much more clearly present
# roughly we need a way to get 4 0's, 2 1's, 1 2, and an arbitrary 8th value, out of 2x2x2 indices
#   ok usually that's ummmmm
#       dims are full/sub
#   see with 1d you would want 1x 0/1
#   with 2d you want 2x 0/2 and 1x 1/1
#   then with 3d you want 4x 0/3, 2x 1/2, and 1x 2/1
#   we could consider dimension n to be associated with 2**n slabs that have n short dimensions to them and are full in other dimensions
#   if think in terms of shortdims its more intuitive.
#   1d has 1 shortdim, that's it.
#   a 2d item introduces 2 elements with 2 shortdims
#   so if you were iterating a 2x2 rect, you could skip 0, then have [0,1] be a 2x1 slab, then [1,0] and [1,1] both be 1x1 slabs.
#   3x3 iteration:
#   [0,0,0]: skipped
#   [0,0,1]: 2x2x1 slab
#   [0,1,0]: 2x1x1 slab
#   [0,1,1]: 2x1x1 slab
#   [1,0,0]: 1x1x1 slab
#   the rest are 1x1x1 slabs, for a total of 1 deepdims=2, 2 deepdims=1, and 4 deepdims=0 :)
#   so each item ummmmmmm let's chart it again
#
# 0 - 1  - - -  4 - 5
# - - -  - - -  - - -
# 2 - 3  - - -  6 - 7
#
# 0: skipped
# 1: [0,0,1]: this could be the 3x3x1 biggest volume ---/---/---
# 2: [0,1,0]: this could be the 3x1x1 01---23
# 3: [0,1,1]: this could be the 3x1x1 45---67
# 4: [1,0,0]: this could be the 1x1x1 0-1
# 5: [1,0,1]: this could be the 1x1x1 4-5 or 2-3
# 6: [1,1,0]: 2-3 or 4-5
# 7: [1,1,1]: 6-7
#
# maybe a different ordering could be clearer.
# it seems like whether or not there is a 1 in an extreme dimension indicates the size
# so the leftmost or rightmost one
# the others indicate positioning :s maybe
#
# > 2: [0,1,0]: this could be the 3x1x1 01---23
# > 3: [0,1,1]: this could be the 3x1x1 45---67
# this middle data could reveal something
# if we assume we can figure it's middle data from say [0,1,...]
# the remaining bit indicates the position perpendicular to the axis of expansion
# so if it is big on axis 0 (here), the 1/0 (coincidentalyl axis 2 here) is indicating the position in axis 2.
# with [0,1,0] if we are considering leftmost or rightmost zeros (which could maybe be detected with a > check against the index ...
## what about considering unflattened 
#   the difference, re unflattened, relates to the permutation of the indices.
#   when the indices are seen with coordinates in the outer dimension, one can see two sets of pairs:
#       00 01
#       11 01
#   when the indices are seen with coordinates in the inner dimension such that each pair is a full index, it is more familiar, but less useful here:
#       00 10
#       01 11
#   compressed: 00 01 10 11
# what is 3d like
#   3d, coords in outer dimension:
#       00 11  00 00  01 01
#       00 11  11 11  01 01
#   3d, coords in inner dimension:
#       000 010  100 110  
#       001 011  101 111
#   compressed: 000 001 010 011 100 101 110 111
#       [reordered above with np.indices rather than xp.meshgrid to have consistent order]
#   i'm kind of seeing that for each location we have a value for each dimension
#   our size on each dimension may actually depend on this value
#   if we consider it spreading. we have more small things than bigger things, so if we consider some dimension being set (or unset)
#   making _all_ dimensions of the empty shape that are >= or <= to it small, i think then it works.
#       so like if our dimension #3 is set, then all dimensions [:3+1] or [3:] or such would become small
#       ok !
#
#   ok i added ray_dimensions to follow that pattern to set sizes, but a little more is needed
#   . gotta decide on what bounds to give them. unsure how intuitive the endpoitns are. 
#   some things are in the middle, some things on the edge.
#   the biggest slab is in the middle.
#   but then nothing else is in the middle in that dimension ..
#   in the next dimension, that dimension that was placed before, is alternated. it takes the value of one of the coordinates.
#   when the coordinate is low, it spreads in one manner, when it is high it spreads in a different manner
#
# 0 - 1  - - -  4 - 5
# - - -  - - -  - - -
# 2 - 3  - - -  6 - 7
#
# 1 [0,0,1] ---/---/--- [:,:,lo:hi]
# 2 [0,1,0] 01---23     [:,lo:hi,:lo]
# 3 [0,1,1] 45---67     [:,lo:hi,hi:]
# 4 [1,0,0] 0 - 1       [lo:hi,:lo,:lo]
# 5 [1,0,1] 2 - 3       [lo:hi,:lo,hi:]
# 6 [1,1,0] 4 - 5       [lo:hi,hi:,:lo]
# 7 [1,1,1] 6 - 7       [lo:hi,hi:,hi:]
#
# there we go. after the first 1, the following values set whether the slice is :lo or hi: to fill the construct


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

    ndlist2b = NDArray(xp.asarray([[1,2],[3,4]]))
    holes = ndlist2b._insert_empty([1,1],[1,1])
    import pdb; pdb.set_trace()
