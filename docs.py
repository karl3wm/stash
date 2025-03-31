import _xp_docs as docs
import sys

def inject_docs(xp):
    for fn, docf in docs.__dict__.items():
        try:
            getattr(xp, fn).__doc__ = docf.__doc__
        except AttributeError:
            pass
        except TypeError:
            pass
    try:
        array = type(xp.asarray([]))
    except AttributeError:
        pass
    else:
        for attrn, docattr in docs.array.__dict__.items():
            try:
                getattr(array, attrn).__doc__ = docattr.__doc__
            except AttributeError:
                pass
            except TypeError:
                pass

for xp_modname in [
    'array_api_strict',
    'array_api_compat.common',
    'array_api_compat.cupy',
    'array_api_compat.numpy',
    'array_api_compat.torch',
]:
    if xp_modname in sys.modules:
        inject_docs(sys.modules[xp_modname])
if 'numpy' in sys.modules:
    inject_docs(sys.modules['numpy'])
