from pathlib import Path
import os
from setuptools import setup, Extension, find_packages

try:
    import pybind11
except Exception as e:
    raise RuntimeError("pybind11 is required to build acornpy: pip install pybind11") from e

here = Path(__file__).resolve().parent
acorn_dir = here.parent  # .../acorn
project_root = acorn_dir.parent  # .../rwalks-reproduce-v2

# Allow overrides via environment variables
faiss_include_dir = os.environ.get("ACORN_FAISS_INCLUDE_DIR")
faiss_lib_dir = os.environ.get("ACORN_FAISS_LIB_DIR")

include_dirs = [
    str(acorn_dir),  # so <faiss/...> resolves to acorn/faiss/...
]

if faiss_include_dir:
    include_dirs.append(faiss_include_dir)

# Typical build output search locations
candidate_lib_dirs = [
    str(acorn_dir / "build/faiss"),
    str(acorn_dir / "build"),
]

library_dirs = []
rpaths = []

if faiss_lib_dir:
    library_dirs.append(faiss_lib_dir)
    rpaths.append(faiss_lib_dir)

for d in candidate_lib_dirs:
    if os.path.isdir(d):
        library_dirs.append(d)
        rpaths.append(d)

extra_compile_args = ["-std=c++17", "-O3", "-fopenmp"]
extra_link_args = ["-fopenmp"] + [f"-Wl,-rpath,{rp}" for rp in rpaths]

ext_modules = [
    Extension(
        name="acornpy._acorn",
        sources=[str(here / "src/pyacorn.cpp")],
        include_dirs=include_dirs + [pybind11.get_include()],
        library_dirs=library_dirs,
        libraries=["faiss"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="acornpy",
    version="0.1.0",
    description="Python bindings for ACORN (Faiss-based ANN with filtering)",
    author="ACORN",
    packages=find_packages(where=str(here)),
    package_dir={"": str(here)},
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
)
