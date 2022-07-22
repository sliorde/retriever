import random
import sys
from copy import deepcopy


def have(x):
    return x is not None


def dont_have(x):
    return not have(x)


def get(obj, *k):
    """
    get object from nested object (list/tuple/dict) using a multi-level key, and return `None` if can't.
    """
    if len(k) == 1:
        k = k[0]
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return obj.get(k, None)
        elif isinstance(obj, (list, tuple)):
            assert isinstance(k, int)
            return obj[k] if k in range(-len(obj), len(obj)) else None
        else:
            raise TypeError
    else:
        return get(get(obj, k[0]), *k[1:])


def sett(obj, value, *k):
    """
    sets the value of a nested object (list/tuple/dict), using a multi-level key,
    creating object down the hierarchy as needed.
    """
    next_obj = get(obj, k[0])
    if dont_have(next_obj) or not isinstance(next_obj, (dict, list)):
        if isinstance(obj, list):
            assert isinstance(k[0], int)
            assert k[0] >= -len(obj)
            for _ in range(len(obj), k[0] + 1):
                obj.append(None)
        if len(k) == 1:
            obj[k[0]] = value
        else:
            if isinstance(k[1], str):
                obj[k[0]] = dict()
            elif isinstance(k[1], int):
                obj[k[0]] = list()
            else:
                raise TypeError
            sett(obj[k[0]], value, *k[1:])


class ModuleWithCache:
    """
    an attempt to create a template for objects (especially torch nn.Modules, especially transformers...)
    that require caching, e.g. for sequential autoregressive sampling. it allows caching intermediate
    calculation results (such as attention keys and values), so that they can be quickly reused for the
    sampling of the next element.
    The idea is that we don't want to carry a "cache" argument and output between all the forward calls
    (since this is ugly). instead, a Module can inherit from the current class, which adds to the
    module a state holding the cache. the cache can be easily disabled, reset, and multiple different
    cache versions can be managed at the same time (each with a different key).
    a Module which is a ModuleWithCache and has children modules which are also ModuleWithCache, needs
    to explicitly call `register_modules_with_cache()` on them.
    see usage example in `test.py`.
    """

    def __init__(self):
        self.cache = dict()
        self._current_key = None
        self._using = False
        self._updating = False

        self._modules_with_cache = []

        self.change_cache_key(0)

    def using_cache(self, yes=True):
        self._using = yes
        for m in self._modules_with_cache:
            m.using_cache(yes)
        return self

    def dont_use_cache(self):
        self.using_cache(False)
        return self

    def updating_cache(self, yes=True):
        self._updating = yes
        for m in self._modules_with_cache:
            m.updating_cache(yes)
        return self

    def dont_update_cache(self):
        self.updating_cache(False)
        return self

    def reset_cache(self):
        self.cache[self._current_key] = dict()
        for m in self._modules_with_cache:
            m.reset_cache()
        return self

    def change_cache_key(self, key=None):
        if key is None:
            while (key is None) or (key in self.cache):
                key = random.randint(0, sys.maxsize)
        if key not in self.cache:
            self.cache[key] = dict()
        self._current_key = key
        for m in self._modules_with_cache:
            m.change_cache_key(key)
        return self

    def register_modules_with_cache(self, m):
        if isinstance(m, ModuleWithCache):
            self._modules_with_cache.append(m)
        elif isinstance(m, (list, tuple, set, dict)):
            if isinstance(m, (dict,)):
                m = m.values()
            for mm in m:
                self.register_modules_with_cache(mm)

    def copy_cache(self, key, new_key, change=False):
        self.cache[new_key] = deepcopy(self.cache[key])
        if change:
            self.change_cache_key(new_key)
        for m in self._modules_with_cache:
            m.copy_cache(key, new_key, change)

    def remove_cache_key(self, key=None):
        for m in self._modules_with_cache:
            m.remove_cache_key(key)
        if key is None:
            key = next(iter(self.cache))
        assert key != self._current_key
        self.cache.pop(key)

    def get_from_cache(self, *k):
        return get(self.cache[self._current_key], *k)

    def update_cache(self, value, *k):
        if self._updating:
            sett(self.cache[self._current_key], value, *k)
