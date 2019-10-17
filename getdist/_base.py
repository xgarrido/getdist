import re
import warnings

def _convert_camel(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


class _BaseObject(object):
    # Compatibility of pep_8_style and camelCase for backwards compatibility

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            try:
                res = object.__getattribute__(self, _convert_camel(name))
            except AttributeError:
                pass
            else:
                warnings.warn("%s is deprecated, use %s" % (name, _convert_camel(name)), DeprecationWarning)
                return res
        # generate standard error
        return object.__getattribute__(self, name)