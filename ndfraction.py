
# Unfortunately, large ndarrays of int64 fractions very quickly exceed
# 64 bits when accumulation operations are run such as matmul.

# A quick workaround could be to output float64 for accumulations.

# This approach tries out reusing fractions.Fraction .
# It looks appropriate to reimplement some operators.

class NDFraction:
    def __init__(self, numerator, denominator=None, *, _normalize=True, _xp=None):
        if type(numerator) is type(self):
            self.numerator = numerator.numerator
            self.denominator = numerator.denominator
            self.xp = numerator.xp
            return
        if _xp is None:
            try:
                xp = self.xp = numerator.__array_namespace__()
            except AttributeError:
                import pdb; pdb.set_trace()
                "well i guess we could extract xp from the parent stack as a cludge"
        else:
            xp = self.xp = _xp
        assert not xp.isdtype(numerator.dtype, 'complex floating')
        if denominator is None:
            denominator = xp.asarray(1)
        numerator, denominator = xp.broadcast_arrays(numerator, denominator)
        if xp.isdtype(numerator.dtype, 'real floating'):
            numerator, denominator = xp.unstack(
                xp.asarray([
                    float(v).as_integer_ratio()
                    for v in xp.unstack(n)
                ]).T.view(2,*numerator.shape)
            )
            self /= type(self)(denominator)
        if _normalize:
            g = ndmath.gcd(numerator, denominator)
            numerator = numerator // g
            denominator = denominator // g
        self.shape = list(numerator.shape)
        self.numerator = numerator
        self.denominator = denominator
        if xp.any(denominator == 0):
            raise ZeroDivisionError(self)
    @property
    def _numerator(self):
        return ComparableArray(self.numerator)
    @property
    def _denominator(self):
        return ComparableArray(self.denominator)
    @property
    def dtype(self):
        return self.numerator.dtype
    @property
    def device(self):
        return self.numerator.device

    def __getitem__(self, idcs):
        return type(self)(self.numerator[idcs], self.denominator[idcs], _normalize=False, _xp=self.xp)

    def astype(x, dtype, *, copy = True, device = None):
        xp = x.xp
        if xp.isdtype(dtype, x.numerator.dtype) and not copy:
            return x
        num = xp.astype(x.numerator, dtype, device=device)
        den = xp.astype(x.denominator, dtype, device=device)
        if xp.isdtype(dtype, 'real floating'):
            return num / den
        else:
            return type(x)(num, den, _normalize=False)

    def _matmul(a, b):
        raise NotImplementedError()

    def sum(self, *params, **kwparams):
        # the problem with this solution is that the product easily overflows int64
        den = self.xp.prod(self.denominator, *params, **kwparams)
        num = self.numerator * den // self.denominator
        num = self.xp.sum(num, *params, **kwparams)
        return type(self)(num, den, _normalize=True)

def gcd2(xp, a, b):
    a_mod_b_ = a % b
    a = xp.asarray(b[...], copy=True)
    if xp.any(a_mod_b_):
        b = a_mod_b_
        mask = a_mod_b_ != 0
        b_ = a_mod_b_[mask]
        a_mod_b_ = a[mask] % b_
        a[mask] = b_
        while xp.any(a_mod_b_):
            b[mask] = a_mod_b_
            mask_ = a_mod_b_ != 0
            mask[mask] = mask_
            a_ = b_[mask_]
            b_ = a_mod_b_[mask_]
            a_mod_b_ = a_ % b_
            a[mask] = b_
    return a

# this clones the Fraction code replacing Fraction with NDFraction
import fractions, types
class ComparableArray:
    '''Compares like a scalar for re-use of scalar functions.'''
    def __init__(self, array):
        self.xp = array.__array_namespace__()
        self.array = array
    def __getitem__(self, idcs):
        return self.array[idcs]
    def __str__(self):
        return self.array.__str__()
    def __repr__(self):
        return self.array.__repr__()
    def __eq__(a, b):
        assert b == 1
        return a.xp.all(a.array == b)
    def __gt__(a, b):
        assert b == 1
        return a.xp.any(a.array > b)
    def __lt__(a, b):
        assert b == 0
        return a.xp.all(a.array < b)
def __ComparableArrayOp(opname):
    return lambda a, b: getattr(a.array, opname)(b)
for opname in [
        f'__{prefix}{opname}__'
        for prefix in ['','r','i']
        for opname in ['floordiv','mod','divmod']
]:
    setattr(ComparableArray, opname, __ComparableArrayOp(opname))
ComparableArray

class ndmath:
    def gcd(a, b):
        try:
            xp = a.__array_namespace__()
        except AttributeError:
            xp = b.__array_namespace__()
        return ComparableArray(gcd2(xp, a, b))
overridden_globals = fractions.__dict__ | dict(
        Fraction=NDFraction,
        math=ndmath,
    )
overridden_members = {}
for name, member in fractions.Fraction.__dict__.items():
    if hasattr(NDFraction, name) and name[0] != '_':
        continue
    if type(member) is types.FunctionType and member.__globals__ is fractions.__dict__:
        new_member = types.FunctionType(
            member.__code__,
            overridden_globals,
            member.__name__,
            member.__defaults__,
            member.__closure__
        )
        overridden_members[member] = new_member
        setattr(NDFraction, name, new_member)
for member in overridden_members.keys():
    closure = member.__closure__
    if closure is not None:
        for cell in member.__closure__:
            new_member = overridden_members.get(cell.cell_contents)
            if new_member is not None:
                cell.cell_contents = new_member


if __name__ == '__main__':
    import array_api_strict as xp_
    import math, random
    import numpy as np

    gcd_test = np.random.randint(1, 10000, [2,2000*1000])
    # gcd_math done on np data rather than xp for speed
    gcd_math = xp_.asarray([math.gcd(gcd_test[0,idx],gcd_test[1,idx]) for idx in range(gcd_test.shape[1])])
    gcd_test = xp_.asarray(gcd_test)
    gcd_ndmath = gcd2(xp_, gcd_test[0,...], gcd_test[1,...])
    assert xp_.all(gcd_math == gcd_ndmath)

    frac = NDFraction(gcd_test[0,...], gcd_test[1,...])
    assert frac.sum().astype(xp_.float32) == xp_.sum(frac.astype(x_.float32))
    #frac1 = NDFraction(xp_.asarray([[1,2],[3,4]]))
    #frac2 = NDFraction(xp_.asarray([[5,6],[7,8]]))
    #print(frac1*frac2)
    import pdb; pdb.set_trace()
    #frac3 = frac1 @ frac2
    #print(frac3)
