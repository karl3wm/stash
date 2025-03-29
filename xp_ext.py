import ctypes
import dlpack # python3 -m pip install pydlpack
  # note: dlpack can also import python buffers as tensors

# The DeviceInfo.has_dtype flags for standard dtypes in this file would be even more useful with nonstandard ones such as bfloat16!
# They are presently set in DeviceInfo.__init__ but could also respond to the std= kwparam of DTypeInfo

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
    def __init__(self, capsule, device_type_id_tuple, **kwparams):
        self.capsule = capsule
        self.device_type_id = device_type_id_tuple
        self.kwparams = kwparams
    def __dlpack_device__(self):
        return self.device_type_id
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
        self.dlpack = dlpack.DLDataType.from_buffer_copy(
            dl_tensor(
                xp.asarray([], dtype=dtype)
                    .__dlpack__(**xp_info(xp).dlpack_kwparams)
            ).dtype
        )
        self.std = std
        self.xp = xp
_dtype_to_info = {}
def dtype_info(dtype, *, xp):
    '''Returns a DTypeInfo object for dtype, containing attributes about the dtype.'''
    info = _dtype_to_info.get(dtype)
    if info is None:
        xp_info = xp_info(xp)
        return _dtype_to_info.get(dtype) or DTypeInfo(dtype, xp=xp)
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
            import warnings
            self.has_writeable_from_dlpack = False
            if xp.__name__ in ['numpy', 'array_api_strict']:
                warnings.warn("numpy isn't writing through dlpack arrays; you can install https://github.com/numpy/numpy/pull/28600 with python3 -m pip install git+https://github.com/karl3wm/numpy@writeable-from-dlpack if has_writeable_from_dlpack is needed")
        else:
            self.has_writeable_from_dlpack = True

        self.devices = [device_info(device, xp=xp) for device in self.info.devices()]
        self.default_device = device_info(self.info.default_device(), xp=xp)

        if xp.__name__ in ['numpy', 'array_api_strict']:
            import numpy as np
            self.backend_array = self.__backend_array_array_api_strict
            self.backend_api = np
            self.is_numpy = True
        else:
            self.backend_array = self.__backend_array_default
            self.backend_api = xp
            self.is_numpy = False

    @staticmethod
    def __backend_array_array_api_strict(array):
        '''Returns the numpy array object underlying an array_api_strict array object.'''
        return array._array
    @staticmethod
    def __backend_array_default(tensor):
        '''Attempts to return the underlying tensor used by the backend API, defaulting to returning the passed tensor.'''
        return tensor

def xp_info(xp):
    '''Returns an XPInfo object for xp, containing attributes about the api.'''
    return _xp_to_info.get(xp) or XPInfo(xp)

def as_nocopy(a, *, xp, shape=None, dtype=None, strides=None):
    '''Alias the data underlying an array as different shape, dtype, and/or strides.
       This presently uses dlpacks, so the return value is only writeable if xp_info(xp).has_writeable_from_dlpack == True
    '''
    info = xp_info(xp)
#    if not info.has_writeable_from_dlpack and info.is_numpy:
#    else:
    if True:
        dlpack_kwparams = info.dlpack_kwparams
        dlpack = a.__dlpack__(**dlpack_kwparams)
        dlt = dl_tensor(dlpack)
        if dtype is not None:
            dlpack_dtype = dtype_info(dtype, xp=xp).dlpack
            if dlpack_dtype.bits != dlt.dtype.bits and shape is None:
                raise ValueError("The new dtype is a different size from the old. Specify the shape.") # It would be intuitive to grow the smallest (i.e. dense) dimension.
            dlt.dtype = dlpack_dtype
        if shape is not None:
            if strides is None and dlt.strides is not None:
                raise ValueError("The passed array is not dense row-major. Specify both shape and strides.")
            dlt.shape = (ctypes.c_long * len(shape))(*shape)
        if strides is not None:
            dlt.strides = (ctypes.c_long * len(strides))(*strides)
    return xp.from_dlpack(forward_dlpack(dlpack, (dlt.device.device_type, dlt.device.device_id), **dlpack_kwparams))

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
    astart = data_ptr(a, xp=xp)
    bstart = data_ptr(b, xp=xp)
    astride = strides_bytes(a, xp=xp)
    bstride = strides_bytes(b, xp=xp)
    astride, adim = max([[astride[idx],idx] for idx in range(astride.shape[0])])
    bstride, bdim = max([[bstride[idx],idx] for idx in range(bstride.shape[0])])
    aend = astart + a.shape[adim] * astride
    bend = bstart + b.shape[bdim] * bstride
    return aend > bstart and bend > astart
