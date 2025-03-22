# NOTE: LIMBS are the mp term for WORDS. They mean basically the same thing.

WANT_ASSERT = True
LIMB_BITS = 64
NAIL_BITS = 0
NUMB_BITS = LIMB_BITS - NAIL_BITS
NUMB_MASK = ((1<<LIMB_BITS)-1) >> NAIL_BITS
NUMB_MAX = NUMB_MASK
NAIL_MASK = ((1<<LIMB_BITS)-1) ^ NUMB_MASK

def ASSERT_CARRY(expr):
    assert expr != 0
def ASSERT_NOCARRY(expr):
    assert expr == 0

class ndmpn:
    def FILL(dst, n, f):
        __dst = dst
        __n = n
        assert __n > 0
        __dst._mp_d[...,:__n] = f
    def ZERO(dst, n):
        assert n >= 0
        if n != 0:
            dst.FILL(n, 0)
    def NORMALIZE(DST):
        xp = DST._xp
        while DST._size > 0:
            if xp.any(DST._mp_d[...,DST._size-1]):
                break
            DST._size -= 1
    def NORMALIZE_NOT_ZERO(DST):
        xp = DST._xp
        while True:
            assert DST._size >= 1
            if xp.any(DST._mp_d[...,DST._size-1]):
                break
            DST._size -= 1
    def STRIP_LOW_ZEROS_NOT_ZERO(ptr):
        '''Strip least significant zero limbs from ptr by incrementing ptr
        and decrementing size.  The number in ptr must be non-zero, ie.
        size!=0 and somewhere a non-zero limb.'''
        assert ptr._size >= 1
        xp = ptr._xp

        while not xp.any(ptr._mpd_d[...,0]):
            ptr._size -= 1
            assert ptr._size >= 1
            ptr._mpd_d = ptr._mpd_d[...,1:]

    def ASSERT_ALWAYS(ptr):
        # let whole loop go dead when no nails
        if NAIL_BITS != 0:
            xp = ptr._xp
            # Check that the nail parts are zero.'
            __nail = ptr._mp_d[...,:ptr._size] & ptr.NAIL_MASK
            assert not xp.any(__nail)
    if WANT_ASSERT:
        def ASSERT(ptr):
            ptr.ASSERT_ALWAYS()
        def ASSERT_ZERO_P(ptr):
            xp = ptr._xp
            assert ptr._size >= 0
            assert not xp.any(ptr._mp_d[...,:ptr._size])
        def ASSERT_NONZERO_P(ptr):
            xp = ptr._xp
            assert ptr._size >= 0
            assert xp.any(ptr._mp_d[...,:ptr._size])
    else:
        def ASSERT(ptr):
            pass
        def ASSERT_ZERO_P(ptr):
            pass
        def ASSERT_NONZERO_P(ptr):
            pass

    def com(d, s, n):
        __d = d._mp_d
        __s = s._mp_d
        __n = n
        assert __n >= 1
        __d[...,:__n] = __s[...,:__n] ^ NUMB_MASK

    def LOGOPS_N(rp, up, vp, n, operation):
        __up = up._mp_d
        __vp = vp._mp_d
        __rp = rp._mp_d
        __n = n
        assert __n > 0
        __rp[...,:__n] = operator(__up[...,:__n], __vp[...,:__n])

    def and_n(rp, up, vp, n):
        rp._mp_d[...,:n] = up._mp_d[...,:n] & vp._mp_d[...,:n]

    def andn_n(rp, up, vp, n):
        rp._mp_d[...,:n] = up._mp_d[...,:n] & ~vp._mp_d[...,:n]

    def nand_n(rp, up, vp, n):
        rp._mp_d[...,:n] = NUMB_MASK^(up._mp_d[...,:n] & vp._mp_d[...,:n])

    def ior_n(rp, up, vp, n):
        rp._mp_d[...,:n] = up._mp_d[...,:n] | vp._mp_d[...,:n]

    def iorn_n(rp, up, vp, n):
        rp._mp_d[...,:n] = up._mp_d[...,:n] | (NUMB_MASK^vp._mp_d[...,:n])

    def nior_n(rp, up, vp, n):
        rp._mp_d[...,:n] = NUMB_MASK^(up._mp_d[...,:n] | vp._mp_d[...,:n])

    def xor_n(rp, up, vp, n):
        rp._mp_d[...,:n] = up._mp_d[...,:n] ^ vp._mp_d[...,:n]

    def xnor_n(rp, up, vp, n):
        rp._mp_d[...,:n] = NUMB_MASK^(up._mp_d[...,:n] ^ vp._mp_d[...,:n])

    def incr_u(p, incr):
        xp = p._xp
        offset = 1
        d = p._mp_d
        __x = d[...,0] + incr
        assert incr <= NUMB_MAX
        d[...,0] = __x & NUMB_MASK
        mask = (__x >> NUMB_BITS) != 0
        while xp.any(mask):
            __x = (d[mask, offset] + 1) & NUMB_MASK
            d[mask, offset] = __x
            mask[mask] = (__x == 0)
            offset += 1
    def decr_u(p, incr):
        xp = p._xp
        offset = 1
        d = p._mp_d
        __x = d[...,0] - incr
        d[...,0] = __x & NUMB_MASK
        mask = (__x >> NUMB_BITS) != 0
        while xp.any(mask):
            __x = d[mask, offset]
            d[mask, offset] = (__x - 1) & NUMB_MASK
            mask[mask] = (__x == 0)
            offset += 1
    if WANT_ASSERT:
        def INCR_U(ptr, size, n):
            ASSERT (size >= 1)
            ASSERT_NOCARRY (ptr.add_1 (ptr, size, n))
        def DECR_U(ptr, size, n):
            ASSERT (size >= 1)
            ASSERT_NOCARRY (ptr.sub_1 (ptr, size, n))
    else:
        def INCR_U(ptr, size, n):
            ptr.incr_u (n)
        def DECR_U(ptr, size, n):
            ptr.decr_u (n)

    # invert limb, quotients ...

    def DIVREM_OR_DIVEXACT(rp, up, n, d):
        if BELOW_THRESHOLD (n, DIVEXACT_1_THRESHOLD)):
            ASSERT_NOCARRY (rp.divrem_1 (0, up, n, d))
        else:
            ASSERT (up.mpn_mod_1 (n, d) == 0)
            rp.mpn_divexact_1 (up, n, d)

class ndmpz:
    def __init__(self, data = None, sign = None, *, alloc = None, size = None, shape = None, xp = None):
        if type(mp_d) is ndmpz:
            xp = self._xp = data._xp
            self._shape = data._shape
            self._sign = data._sign
            self._size = data._size
            self._mp_alloc = data._mp_alloc
            self._mp_d = data._mp_d
        else:
            if data is None:
                if size is None:
                    size = 0
                if alloc is None:
                    alloc = size
                if shape is None:
                    shape = sign.shape
                if alloc > 0:
                    data = xp.empty([*shape, alloc], dtype=xp.uint64)
            else:
                if alloc is not None:
                    assert alloc == data.shape[-1]
                else:
                    alloc = data.shape[-1]
                if size is None:
                    size = alloc
                if xp is None:
                    xp = data.__array_namespace__()
            if sign is None:
                sign = xp.empty(shape, dtype=xp.bool)
            else:
                assert sign.shape == shape
            self._xp = xp
            self._shape = shape
            self._sign = sign
            self._size = size
            self._mp_alloc = alloc
            self._mp_d = data

    def abs(u, w=None):
        if u is not w:
            if w is None:
                w = ndmpz(shape = u._shape)
            elif w._shape != u._shape:
                w._shape = u._shape
                w.ALLOC_assign(0)

            size = u.ABSIZ()
            w = w.NEWALLOC(size)

            w._mp_d[...,:size] = u._mp_d[...,:size]
        
        w._sign[...] = 0

    def __aors(FUNCTION, NEGATE):
        def _FUNCTION(u, v, w=None):
            xp = u._xp

            usign = u._sign
            vsign = xp.logical_xor(NEGATE, v._sign)
            abs_usize = u.ABSIZ()
            abs_vsize = v.ABSIZ()

            if abs_usize < abs_vsize:
                # Swap U and V.
                u, v = v, u
                usign, vsign = vsign, usign
                abs_usize, abs_vsize = abs_vsize, abs_usize

            # True: ABS_USIZE >= ABS_VSIZE.

            # If not space for w (and possible carry), increase space.
            wsize = abs_usize + 1
            w = w.REALLOC(wsize)

            # These must be after realloc (u or v may be the same as w).
            up = u._mp_d
            vp = v._mp_d

            if xp.any(xp.bitwise_xor(usign, vsign)):
                # U and V have different sign.  Need to compare them to determine
                # which operand to subtract from which.
                raise NotImplementedError()
            else:
                # U and V have same sign.  Add them.
                raise NotImplementedError()

    def SIZ(x):
        raise NotImplementedError('ndmpz has n-dimensional _sign')
        return x._mp_size
    def SIZ_assign(x, s):
        x._size = s
    def ABSIZ(x):
        return x._size
    def PTR(x):
        raise NotImplementedError('ndmpz has both ._mp_d and ._sign')
        return x._mp_d
    def PTR_assign(x, d):
        raise NotImplementedError('ndmpz has both ._mp_d and ._sign')
        x._mp_d = d
    def ALLOC(x):
        return x._mp_alloc

    def TMP_INIT(X, NLIMBS):
        xp = X._xp
        __x = X
        assert NLIMBS >= 1
        __x._mp_alloc = NLIMBS
        __x._mp_d = xp.empty([*__x._shape, NLIMBS], dtype=xp.uint64)
        __x._mp_s = xp.empty(__x._shape, dtype=xp.bool)
    def REALLOC(z, n):
        return z.realloc(n) if n > z.ALLOC() else z
    def NEWALLOC(z, n):
        return z.newalloc(n) if n > z.ALLOC() else z
    def EQUAL_1_P(z):
        xp = z._xp
        return z.ABSIZ() == 1 and xp.all(z._mp_d[...,0] == 1) and xp.all(z._mp_s[...] == 0)

    def CHECK_FORMAT(z):
        xp = z._xp
        assert z.ABSIZ() == 0 or xp.all(z.PTR[...,z.ABSIZ() - 1] != 0)
        assert z.ALLOC() >= z.ABSIZ()
        if NAIL_BITS != 0:
            ptr, size, xp = z.PTR(), z.ABSIZ(), z._xp
            for __i in range(size):
                __nail = ptr[...,__i] & z.NAIL_MASK
                assert xp.all(__nail == 0)

    def _realloc(m, new_alloc):
        xp = m._xp

        # Never allocate zero space.
        new_alloc = max(new_alloc, 1)

        alloc = m.ALLOC()
        abs_size = m.ABSIZ()

        # i seem to be trying to reconsolidate the logic around here
        # since i don't have to dispatch to an allocator and have an additional concern of the sign

        # i think i should not plan around forcing reallocation here
        #   if i'm having working memory concerns

        # alloc, size, shape
        #   sign vs data; sign only changes on new allocation and shape
        if alloc == 0:
            m._mp_d = xp.empty([*m._shape, new_alloc], dtype=xp.uint64l)
            m._mp_s = m.xp.empty(m._shape, dtype=xp.bool)
            m.ALLOC_assign(new_alloc)
        elif alloc >= new_alloc:
            if abs_size > new_alloc:
                # Don't create an invalid number; if the current value doesn't fit after
                # reallocation, clear it to 0.
                m.SIZ_assign(0)
        else:
            mp_d = xp.empty([*m._shape, new_alloc], dtype=xp.uint64l)
            mp_d[...,:abs_size] = m._mp_d[...,:abs_size]
            m.ALLOC_assign(new_alloc)
            m._mp_d = mp_d
        return m
    _newalloc = _realloc
