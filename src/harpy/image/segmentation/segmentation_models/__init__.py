import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_cellpose': ['cellpose_callable'],
#         '_instanseg': ['instanseg_callable'],
#     }
# )
