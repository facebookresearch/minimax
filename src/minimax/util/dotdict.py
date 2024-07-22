"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self)))


class DefaultDotDict(dict):
    __delattr__ = dict.__delitem__

    def __init__(self, dct, default=None):
        super().__init__(dct)
        self._default = default

    def __getstate__(self):
        return (self, self._default)

    def __setstate__(self, state):
        self.update(state[0])
        self.__dict__ = self
        self._default = state[1]

    def __setattr__(self, name, value):
        if name == "_default":
            super().__setattr__("_default", value)
        else:
            self.__setitem__(name, value)

    def __deepcopy__(self, memo):
        return DefaultDotDict(copy.deepcopy(dict(self)), default=self._default)

    def __getattr__(self, name):
        if name == "_default":
            return self._default
        else:
            try:
                return self.__getitem__(name)
            except:
                return self._default
