"""
This type stub file was generated by pyright.
"""

from .ImageFile import ImageFile

class GdImageFile(ImageFile):
    format: str
    format_description: str
    ...

def open(fp, mode: str = ...): ...