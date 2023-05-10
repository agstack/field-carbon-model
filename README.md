Field-Based Carbon Flux Model
=============================

A model for field-specific carbon flux estimation, based on the NASA Soil Moisture Active Passive (SMAP) Level 4 Carbon (L4C) model.


Installation
-------------------

While this project is in active development, it's currently recommended to install the Python package in "editable" mode, using `pip`:

```sh
# From the directory containing setup.py
pip install -e .
```

During development, it's convenient to rely upon the [`pyl4c` package](https://github.com/arthur-e/pyl4c) for some functionality. However, `pyl4c` is not a hard requirement because it has some dependencies that may be difficult for some users to install (HDF5 and GDAL). To install with support for `pyl4c`:

```sh
# Install GDAL first, ensuring that the version matches the system library
pip install GDAL==$(gdal-config --version)
pip install -e .[pyl4c]
```


Running Tests
-------------

The test suite depends on `pytest`.

```sh
pip install -e .[dev]
```

**Tests can be run with:**

```sh
pytest tests/
```

Some tests depend on having `pyl4c` installed and will be skipped if the module is not available.


Prior Art and Citation
----------------------

Model code here is based heavily on the publicly available [`pyl4c` package](https://github.com/arthur-e/pyl4c) (Endsley et al. 2022). See `REFERENCES` for a complete list of references.
