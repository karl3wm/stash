import ctypes
import dlpack # python3 -m pip install pydlpack
  # note: dlpack can also import python buffers as tensors

# The DeviceInfo.has_dtype flags for standard dtypes in this file would be even more useful with nonstandard ones such as bfloat16!
# They are presently set in DeviceInfo.__init__ but could also respond to the std= kwparam of DTypeInfo

# Some of the stride mutations are awkward; was planning to make strides and shapes all be ndarrays to ease such things.

_DLMTV_p = ctypes.POINTER(dlpack.DLManagedTensorVersioned)
_DLMT_p = ctypes.POINTER(dlpack.DLManagedTensor)
def dl_tensor(capsule):
    '''Returns the DLTensor held by a DLPack capsule returned by array.__dlpack__().'''
    # the dltype functions might be more efficient using raw offsets
    try:
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b'dltensor_versioned')
        dlmt = ctypes.cast(ptr, _DLMTV_p).contents
    except:
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b'dltensor')
        dlmt = ctypes.cast(ptr, _DLMT_p).contents
    return dlmt.dl_tensor

class forward_dlpack:
    '''Converts a DLPack capsule from array.__dlpack__() back into an object that can be consumed by xp.from_dlpack(...).'''
    def __init__(self, capsule, dlpack_device_tuple, **kwparams):
        self.capsule = capsule
        self.dlpack_device = dlpack_device_tuple
        self.kwparams = kwparams
    def __dlpack_device__(self):
        return self.dlpack_device
    def __dlpack__(self, **kwparams):
        assert kwparams == self.kwparams
        return self.capsule

class DTypeInfo:
    def __init__(self, dtype, *, xp, kind=None, std=None):
        assert dtype not in _dtype_to_info and '''duplicate dtype'''
        _dtype_to_info[dtype] = self
        if kind is None:
            for kind in ['bool', 'signed integer', 'unsigned integer', 'real floating', 'complex floating']:
                if xp.isdtype(dtype, kind):
                    break
            else:
                kind = None
        if kind in ['signed integer', 'unsigned integer']:
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
        self.xp = xp_info(xp)
        self.backend = self.xp.asbackend_dtype(self.dtype)
        self.dlpack = dlpack.DLDataType.from_buffer_copy(
            dl_tensor(
                xp.asarray([], dtype=dtype)
                    .__dlpack__(**self.xp.dlpack_kwparams)
            ).dtype
        )
        self.std = std
_dtype_to_info = {}
def dtype_info(dtype, *, xp):
    '''Returns a DTypeInfo object for dtype, containing attributes about the dtype.'''
    info = _dtype_to_info.get(dtype)
    if info is None:
        xp_info(xp) # ensures standard dtypes are constructed with standard kinds and names
        return _dtype_to_info.get(dtype) or DTypeInfo(dtype, xp=xp)
    else:
        return info

class DeviceInfo:
    def __init__(self, device, *, xp):
        _device_to_info[device] = self
        self.device = device
        self.xp = xp_info(xp)
        for dtype_kind in ['bool', 'signed integer', 'unsigned integer', 'real floating', 'complex floating']:
            for dtype_name, dtype in self.xp.info.dtypes(kind=dtype_kind, device=device).items():
                setattr(type(self), 'has_' + dtype_name, False)
                setattr(self, 'has_' + dtype_name, True)
                setattr(self, dtype_name, dtype)
                dtypeinfo = _dtype_to_info.get(dtype)
                if dtypeinfo is None:
                    DTypeInfo(dtype, xp=xp, kind=dtype_kind, std=dtype_name)
                else:
                    assert dtypeinfo.kind == dtype_kind
                    assert dtypeinfo.std is None or dtypeinfo.std == dtype_name
                    dtypeinfo.std = dtype_name
        for dtype_name, dtype in self.xp.info.default_dtypes().items():
            setattr(self, 'default_' + dtype_name.split(' ',1)[0] + '_type', dtype_info(dtype, xp=xp))
        self.backend = self.xp.asbackend_device(self.device)
        try:
            self.dlpack = dlpack.DLDevice.from_buffer_copy(
                dl_tensor(
                    xp.asarray([], device=device)
                        .__dlpack__(**self.xp.dlpack_kwparams)
                ).device
            )
            self.dlpack_device = (self.dlpack.device_type.value, self.dlpack.device_id)
            self.has_dlpack = True
        except RuntimeError:
            self.has_dlpack = False

_device_to_info = {}
def device_info(device, *, xp):
    '''Returns a DeviceInfo object for device, containing attributes about the device.'''
    return _device_to_info.get(device) or DeviceInfo(device, xp=xp)

_xp_to_info = {}
class XPInfo:
    def __init__(self, xp):
        _xp_to_info[xp] = self
        self.xp = xp
        self.info = xp.__array_namespace_info__()

        for capname, value in self.info.capabilities().items():
            capname = capname.replace('-','_').replace(' ','_')
            if type(value) is bool:
                capname = 'has_' + capname
            setattr(self, capname, value)

        self.is_array_api_strict = False
        self.is_array_api_compat = False
        self.is_numpy = False
        self.asbackend = self.__asbackend_default
        self.asbackend_dtype = self.__asbackend_default
        self.asbackend_device = self.__asbackend_default
        self.backend = xp
        if xp.__name__ == 'array_api_strict':
            self.is_array_api_strict = True
            self.asbackend = self.__asbackend_array_api_strict_array
            self.asbackend_dtype = self.__asbackend_array_api_strict_dtype
            self.asbackend_device = self.__asbackend_array_api_strict_device
            self.is_numpy = True
        elif xp.__name__ == 'numpy':
            self.is_numpy = True
        elif xp.__name__.startswith('array_api_compat.'):
            self.is_array_api_compat = True
            if xp.__name__.endswith('.numpy'):
                self.is_numpy = True
        if self.is_numpy:
            import numpy as np
            self.backend = np


        class dlpack_probe:
            def __init__(probe, array):
                probe.array = array
            def __dlpack_device__(probe):
                return (dlpack.DLDeviceType.from_label('DLCPU'), None)
            def __dlpack__(probe, **kwparams):
                self.dlpack_kwparams = kwparams
                return probe.array.__dlpack__(**kwparams)
        test_array = xp.asarray([0])
        test_imported_array = xp.from_dlpack(dlpack_probe(test_array))
        try:
            test_imported_array[0] = 1
            assert test_array[0] == 1
        except:
            self.has_writeable_from_dlpack = False
            if self.is_numpy:
                import warnings
                warnings.warn("numpy isn't writing through dlpack arrays; you can install https://github.com/numpy/numpy/pull/28600 with python3 -m pip install git+https://github.com/karl3wm/numpy@writeable-from-dlpack if has_writeable_from_dlpack is needed")
        else:
            self.has_writeable_from_dlpack = True

        self.devices = [device_info(device, xp=xp) for device in self.info.devices()]
        self.default_device = device_info(self.info.default_device(), xp=xp)

    @staticmethod
    def __asbackend_array_api_strict_array(array):
        '''Returns the numpy array object underlying an array_api_strict array object.'''
        return array._array
    @staticmethod
    def __asbackend_array_api_strict_dtype(dtype):
        '''Returns the numpy dtype object underlying an array_api_strict dtype object.'''
        return dtype._np_dtype
    @staticmethod
    def __asbackend_array_api_strict_device(device):
        '''Returns the numpy device information underlying an array_api_strict device object.'''
        return device._device
    @staticmethod
    def __asbackend_default(value):
        '''Attempts to return the underlying value used by the backend API, defaulting to returning the passed value.'''
        return value

def xp_info(xp):
    '''Returns an XPInfo object for xp, containing attributes about the api.'''
    return _xp_to_info.get(xp) or XPInfo(xp)

def as_nocopy(a, *, xp, shape=None, dtype=None, strides_bytes=None):
    '''Alias the data underlying an array as different shape, dtype, and/or strides.'''
    info = xp_info(xp)
    dtype_old = dtype_info(a.dtype, xp=xp)
    if dtype is None:
        dtype_new = dtype_old
    else:
        dtype_new = dtype_info(dtype, xp=xp)
        if dtype_old.bits != dtype_new.bits and shape is None and strides is None:
            raise ValueError("The new dtype is a different size from the old. Specify the shape.") # It would be intuitive to grow the smallest (i.e. dense) dimension.
    if not info.has_writeable_from_dlpack and info.is_numpy:
        a = info.asbackend(a)
        np = info.backend
        buffer = np.data
        if shape is not None:
            if strides_bytes is None:
                if buffer.C_CONTIGUOUS:
                    strides_bytes = shape_to_strides_row_major(shape, elem_size=dtype_new.size, xp=xp)
                elif buffer.F_CONTIGUOUS:
                    strides_bytes = shape_to_strides_column_major(shape, elem_size=dtype_new.size, xp=xp)
                else:
                    raise ValueError("The passed array is not dense. Specify both shape and strides.")
        else:
            shape = a.shape
            if strides_bytes is None:
                strides_bytes = a.strides
        a = np.frombuffer(a.data, dtype=dtype_new.backend)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides_bytes, subok=True)
    else:
        dlpack_kwparams = info.dlpack_kwparams
        dlpack = a.__dlpack__(**dlpack_kwparams)
        dlt = dl_tensor(dlpack)
        dlt.dtype = dtype_new.dlpack
        if shape is not None:
            if strides_bytes is None and dlt.strides is not None:
                raise ValueError("The passed array is not dense row-major. Specify both shape and strides.")
            dlt.shape = (ctypes.c_long * len(shape))(*shape)
        if strides_bytes is not None:
            dlt.strides = (ctypes.c_long * len(strides))(*[stride//dtype_new.size for stride in strides_bytes])
        return xp.from_dlpack(forward_dlpack(dlpack, (dlt.device.device_type, dlt.device.device_id), **dlpack_kwparams))

def shape_to_strides_row_major(shape, *, elem_size=1, xp):
    tmp = xp.asarray(shape, copy=True)
    tmp[1:] = tmp[:0:-1]
    tmp[0] = elem_size
    return xp.cumulative_prod(tmp)[::-1]

def shape_to_strides_column_major(shape, *, elem_size=1, xp):
    tmp = xp.asarray(shape, copy=True)
    tmp[1:] = tmp[:-1]
    tmp[0] = elem_size
    return xp.cumulative_prod(tmp)

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
        return shape_to_strides_row_major(a.shape, elem_size=elem_size, xp=xp)

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
    astart = data_ptr(a, xp=xp)
    bstart = data_ptr(b, xp=xp)
    astride = strides_bytes(a, xp=xp)
    bstride = strides_bytes(b, xp=xp)
    astride, adim = max([[astride[idx],idx] for idx in range(astride.shape[0])])
    bstride, bdim = max([[bstride[idx],idx] for idx in range(bstride.shape[0])])
    aend = astart + a.shape[adim] * astride
    bend = bstart + b.shape[bdim] * bstride
    return aend > bstart and bend > astart
