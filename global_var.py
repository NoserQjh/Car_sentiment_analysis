# -*- coding: utf-8 -*-
class GLOBAL_VAR:
    def __init__(self):
        # freeze_support()
        # manager=Manager()
        # self.manager=manager
        self._global_dict = dict()
    def set_value(self, name, value):
        self._global_dict[name] = value

    def get_value(self, name, default_value=None):
        try:
            return self._global_dict[name]
        except KeyError:
            return default_value

gl = GLOBAL_VAR()