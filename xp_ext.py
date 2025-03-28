import ctypes
import dlpack # python3 -m pip install pydlpack
  # note: dlpack can also alias python buffers as tensors

_DLMTV_p = ctypes.POINTER(dlpack.DLManagedTensorVersioned)
_DLMT_p = ctypes.POINTER(dlpack.DLManagedTensor)
def dl_tensor(capsule):
    '''Returns the DLTensor held by a DLPack capsule returned by array.__dlpack__().'''
    # the dltype functions could be made more efficient by using raw offsets
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b'dltensor')
    dlmt = ctypes.cast(ptr, _DLMTV_p)
    version = dlmt.contents.version
    if version.major + version.minor > 10000:
        dlmt = ctypes.cast(ptr, _DLMT_p)
    return dlmt.contents.dl_tensor

_dtype_to_dldatatype = {}
def dtype_to_dldatatype(dtype, *, xp):
    '''Converts an xp data type to the equivalent data used by a DLPack.'''
    dltype = _dtype_to_dldatatype.get(dtype)
    if dltype is None:
        dlp = xp.asarray([], dtype=dtype).__dlpack__()
        dltype = dlpack.DLDataType.from_buffer_copy(dl_tensor(dlp).dtype)
        _dtype_to_dldatatype[dtype] = dltype
    return dltype

class DTypeInfo:
    def __init__(self, dtype, *, xp, kind=None):
        if kind is None:
            for kind in ['bool', 'signed integer', 'unsigned integer', 'real floating', 'complex floating']:
                if xp.isdtype(dtype, kind):
                    break
            else:
                kind = None
        if kind in ['signed integer, unsigned integer']:
            self.integral = True
            self.floating = False
            self.signed = (kind == 'signed integer')
            self.info = xp.iinfo(dtype)
            self.eps = 1
            self.smallest_normal = 1
        elif kind in ['real floating', 'complex floating']:
            self.integral = False
            self.floating = True
            self.signed = True
            self.info = xp.finfo(dtype)
            self.eps = self.info.eps
            self.smallest_normal = self.info.smallest_normal
        else:
            self.integral = False
            self.floating = False
            self.signed = False
            self.info = None
            if kind == 'bool':
                self.eps = True
                self.smallest_normal = True
                self.bits = 1
                self.size = 1
                self.max = True
                self.min = False
            else:
                self.eps = None
                self.smallest_normal = None
                self.bits = None
                self.size = None
                self.max = None
                self.min = None
        self.kind = kind
        if self.info is not None:
            self.bits = self.info.bits
            self.size = self.bits >> 3
            self.max = self.info.max
            self.min = self.info.min
        self.dtype = dtype
        self.xp = xp
_dtype_to_info = {}
def dtype_info(dtype, *, xp):
    info = _dtype_to_info.get(dtype)
    if info is None:
        # there is more unused information available in xp.__array_namespace_info().dtypes() and .default_dtypes()
        info = DTypeInfo(dtype, xp=xp)
        _dtype_to_info[dtype] = info
    return info

class forward_dlpack:
    '''Converts a DLPack capsule from array.__dlpack__() back into an object that can be consumed by xp.from_dlpack(...).'''
    def __init__(self, capsule, stream=None):
        self.capsule = capsule
        self.stream = stream
    def __dlpack__(self, stream=None):
        assert stream is self.stream
        return self.capsule

def as_nocopy(a, *, xp, shape=None, dtype=None, strides=None):
    '''Alias the data underlying an array as different shape, dtype, and/or strides.
       Note: The current implementation uses DLPack under the hood, which is read-only in NumPy.
    '''
    dlpack = a.__dlpack__()
    dlt = dl_tensor(dlpack)
    if dtype is not None:
        dtype = dtype_to_dldatatype(dtype, xp=xp)
        if dtype.bits != dlt.dtype.bits and shape is None:
            raise ValueError("The new dtype is a different size from the old. Specify the shape.") # It would be intuitive to grow the smallest (i.e. dense) dimension.
        dlt.dtype = dtype
    if shape is not None:
        if strides is None and dlt.strides is not None:
            raise ValueError("The passed array is not dense row-major. Specify both shape and strides.")
        dlt.shape = (ctypes.c_long * len(shape))(*shape)
    if strides is not None:
        dlt.strides = (ctypes.c_long * len(strides))(*strides)
    return xp.from_dlpack(forward_dlpack(dlpack))

def strides_bytes(a, *, xp):
    '''Returns the strides held by an array as the raw bytes between elements for each dimension.'''
    dlpack = a.__dlpack__()
    dlt = dl_tensor(dlpack)
    strides_p = dlt.strides
    elem_size = dlt.dtype.bits >> 3
    if strides_p:
        # this could benefit from a function to convert a data_ptr to an array
        return xp.asarray(strides_p[:dlt.ndim]) * elem_size
    else:
        # tensor is compact and row-majored
        # this could benefit from a backend-specific approach
        tmp = xp.asarray(a.shape)
        tmp[1:] = tmp[:0:-1]
        tmp[0] = elem_size
        return xp.cumulative_prod(tmp)[::-1]

def data_ptr(a, *, xp):
    '''Returns the underlying address as an integer of the first element in an array.'''
    dlpack = a.__dlpack__()
    dlt = dl_tensor(dlpack)
    return dlt.data + dlt.byte_offset

def may_share_memory(a, b, *, xp):
    '''Returns True if and only if the underlying data ranges of a and b overlap.
       Note that this does not necessarily mean that they share memory, as they may be differently strided.
    '''
    if a.device != b.device:
        return False
    astart = data_ptr(a)
    bstart = data_ptr(b)
    astride = strides_bytes(a, xp=xp)
    bstride = strides_bytes(b, xp=xp)
    astride, adim = max([[astride[idx],idx] for idx in range(astride.shape[0])])
    bstride, bdim = max([[bstride[idx],idx] for idx in range(bstride.shape[0])])
    aend = astart + a.shape[adim] * astride
    bend = bstart + b.shape[bdim] * bstride
    return aend > bstart and bend > astart
