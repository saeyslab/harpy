import lazy_loader as lazy

from harpy.utils.pylogger import get_pylogger

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
