import ctypes
import dlpack # python3 -m pip install pydlpack
  # note: dlpack can also alias python buffers for tensor libs that can't

# this could be made more efficient by using raw offsets

_DLMTV_p = ctypes.POINTER(dlpack.DLManagedTensorVersioned)
_DLMT_p = ctypes.POINTER(dlpack.DLManagedTensor)
def dl_tensor(_dlpack):
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(_dlpack, b'dltensor')
    dlmt = ctypes.cast(ptr, _DLMTV_p)
    version = dlmt.contents.version
    if version.major + version.minor > 10000:
        dlmt = ctypes.cast(ptr, _DLMT_p)
    return dlmt.contents.dl_tensor

def strides_bytes(a, *, xp):
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

def data_ptr(a):
    dlpack = a.__dlpack__()
    dlt = dl_tensor(dlpack)
    return dlt.data + dlt.byte_offset

def may_share_memory(a, b, *, xp):
    if a.device != b.device:
        return False
    astart = data_ptr(a)
    bstart = data_ptr(b)
    astride = strides_bytes(a, xp=xp)
    bstride = strides_bytes(b, xp=xp)
    astride, adim = max([[astride[idx],idx] for idx in range(astride.shape[0])])
    bstride, bdim = max([[bstride[idx],idx] for idx in range(bstride.shape[0])])
    aend = astart + astride
    bend = bstart + bstride
    if aend > bstart and bend > astart:
        import pdb; pdb.set_trace()
    return aend > bstart and bend > astart

#def astype_nocopy(a, type, *, xp):
#    dlpack = a.__dlpack__()
#    dlt = dl_tensor(dlpack)
#    # i need a way to convert the dtype.
#    # if i use .from_dlpack then it would from xp to dlpack.
#    xp.from_dlpack(dlpack)
