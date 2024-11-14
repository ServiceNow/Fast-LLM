import sys

try:
    import pybind11
    import setuptools
except ImportError:
    raise ImportError(
        "Could not import third party module during setup."
        " Please make sure it is installed before installing Fast-LLM, and use `--no-build-isolation"
    )

# Minimum setuptools version required to parse setup.cfg metadata.
_SETUPTOOLS_MIN_VERSION = "30.3"

if setuptools.__version__ < _SETUPTOOLS_MIN_VERSION:
    print(f"Error: setuptools version {_SETUPTOOLS_MIN_VERSION} " "or greater is required")
    sys.exit(1)

cpp_extension = setuptools.Extension(
    "fast_llm.csrc.data",
    sources=["fast_llm/csrc/data.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-Wall", "-shared", "-std=c++11", "-fPIC"],
    extra_link_args=["-fdiagnostics-color"],
    language="c++",
)

setuptools.setup(
    ext_modules=[cpp_extension],
)
