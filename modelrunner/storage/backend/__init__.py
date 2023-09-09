from .memory import MemoryStorage

AVAILABLE_STORAGE = [MemoryStorage]

try:
    from .hdf import HDFStorage
except ImportError:
    pass  # h5py package not installed
else:
    AVAILABLE_STORAGE.append(HDFStorage)

try:
    from .zarr import ZarrStorage
except ImportError:
    pass  # zarr package not installed
else:
    AVAILABLE_STORAGE.append(ZarrStorage)
