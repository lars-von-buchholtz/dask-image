# -*- coding: utf-8 -*-

__author__ = """Lars von Buchholtz"""
__email__ = "lars.von.buchholtz@gmail.com"


from ._feature import (
    blob_log,
    blob_dog,
    blob_doh,
    peak_local_max
)


blob_log.__module__ = __name__
blob_dog.__module__ = __name__
blob_doh.__module__ = __name__
peak_local_max.__module__ = __name__
