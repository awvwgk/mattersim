# =============================================================================
# MatterSim + LAMMPS Docker Image
# =============================================================================
# Provides LAMMPS with GPU support (Kokkos + CUDA) and the MatterSim ML
# potential via the MLIAP Python interface. Supports multi-GPU runs via
# MPI + UCX (CUDA-aware).
#
# Build:
#   docker build -t mattersim-lammps --build-arg KOKKOS_ARCH=AMPERE80 .
#
# Supported KOKKOS_ARCH values (one per build):
#   VOLTA70   — V100
#   AMPERE80  — A100 (default)
#   AMPERE86  — RTX A6000, A5000, A4000
#   HOPPER90  — H100, H200
#
# Run:
#   docker run --gpus all -it mattersim-lammps
# =============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools
RUN apt-get update -yq && \
    apt-get upgrade -yq && \
    apt-get install -yq --no-install-recommends \
        cmake make g++ git wget ca-certificates \
        openssh-client \
        libtool autoconf automake pkg-config && \
    rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------------------------
# UCX with CUDA support (for GPU-direct MPI transfers)
# ---------------------------------------------------------------------------
ARG UCX_VERSION=1.19.0
RUN wget -q https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz && \
    tar xzf ucx-${UCX_VERSION}.tar.gz && \
    cd ucx-${UCX_VERSION} && \
    ./configure --prefix=/usr/local/ucx \
                --with-cuda=/usr/local/cuda \
                --enable-mt && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf ucx-${UCX_VERSION}*

ENV LD_LIBRARY_PATH=/usr/local/ucx/lib:${LD_LIBRARY_PATH}
ENV UCX_WARN_UNUSED_ENV_VARS=n


# ---------------------------------------------------------------------------
# OpenMPI with CUDA + UCX support
# ---------------------------------------------------------------------------
ARG OMPI_VERSION=4.1.6
RUN wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OMPI_VERSION}.tar.gz && \
    cd openmpi-${OMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi \
                --with-cuda=/usr/local/cuda \
                --with-ucx=/usr/local/ucx \
                --enable-mca-no-build=btl-uct && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf openmpi-${OMPI_VERSION}*

ENV PATH=/usr/local/openmpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:${LD_LIBRARY_PATH}
# Allow mpirun as root (containers run as root by default)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1


# ---------------------------------------------------------------------------
# Miniconda + MatterSim Python environment
# ---------------------------------------------------------------------------
ARG CONDA_PREFIX=/opt/miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_PREFIX} && \
    rm /tmp/miniconda.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}

# Use conda-forge only (avoids Anaconda ToS acceptance requirement)
RUN conda config --set default_channels '[]' && \
    conda config --remove channels defaults 2>/dev/null; \
    conda config --add channels conda-forge && \
    conda create -n env -c conda-forge --override-channels python=3.12 -y && \
    conda clean --all --yes

# Remove conda plugins that conflict with env's pydantic version
RUN conda remove -n base --force anaconda-auth conda-anaconda-tos -y 2>/dev/null || true

# Put conda env's python on PATH for all subsequent steps
ENV PATH=${CONDA_PREFIX}/envs/env/bin:${PATH}

# Install MatterSim and LAMMPS dependencies
RUN python -m ensurepip && \
    python -m pip install --no-cache-dir uv && \
    uv pip install --no-cache --python $(which python) \
        mattersim \
        cupy-cuda12x \
        cython \
        numpy

# Activate conda env in interactive shells
RUN echo ". ${CONDA_PREFIX}/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate env" >> /etc/bash.bashrc


# ---------------------------------------------------------------------------
# LAMMPS with Kokkos + MLIAP + Python
# ---------------------------------------------------------------------------
ARG LAMMPS_VERSION=patch_30Mar2026
ARG KOKKOS_ARCH=AMPERE80
RUN git clone --depth 1 --branch ${LAMMPS_VERSION} \
        https://github.com/lammps/lammps.git /tmp/lammps && \
    mkdir /tmp/lammps/build && cd /tmp/lammps/build && \
    cmake ../cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local/lammps \
        -DBUILD_SHARED_LIBS=yes \
        -DBUILD_MPI=yes \
        -DPKG_KOKKOS=yes \
        -DKokkos_ENABLE_CUDA=yes \
        -DKokkos_ARCH_${KOKKOS_ARCH}=yes \
        -DPKG_ML-IAP=yes \
        -DMLIAP_ENABLE_PYTHON=ON \
        -DPKG_PYTHON=yes \
        -DCMAKE_CXX_STANDARD=17 \
        -DPython_EXECUTABLE=$(which python) \
        -DPython_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython$(python -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))").so \
        -DPython_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))") \
        -DPython_NumPy_INCLUDE_DIR=$(python -c "import numpy; print(numpy.get_include())") \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    make -j$(nproc) && \
    make install && \
    cd /tmp/lammps/python && python -m pip install . && \
    rm -rf /tmp/lammps


# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
# Torch's bundled NVIDIA libs (cu128) must come before system CUDA (12.6)
# to avoid symbol version mismatches.
ENV NVIDIA_PYTHON_LIBS=${CONDA_PREFIX}/envs/env/lib/python3.12/site-packages/nvidia
ENV PATH=${CONDA_PREFIX}/envs/env/bin:/usr/local/lammps/bin:${PATH}
ENV LD_LIBRARY_PATH=${NVIDIA_PYTHON_LIBS}/cublas/lib:${NVIDIA_PYTHON_LIBS}/cuda_cupti/lib:${NVIDIA_PYTHON_LIBS}/cuda_nvrtc/lib:${NVIDIA_PYTHON_LIBS}/cuda_runtime/lib:${NVIDIA_PYTHON_LIBS}/cudnn/lib:${NVIDIA_PYTHON_LIBS}/cufft/lib:${NVIDIA_PYTHON_LIBS}/curand/lib:${NVIDIA_PYTHON_LIBS}/cusolver/lib:${NVIDIA_PYTHON_LIBS}/cusparse/lib:${NVIDIA_PYTHON_LIBS}/cusparselt/lib:${NVIDIA_PYTHON_LIBS}/nccl/lib:${NVIDIA_PYTHON_LIBS}/nvjitlink/lib:${NVIDIA_PYTHON_LIBS}/nvtx/lib:${CONDA_PREFIX}/envs/env/lib:/usr/local/lammps/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${CONDA_PREFIX}/envs/env/lib/python3.12/site-packages
ENV LAMMPS_BIN=/usr/local/lammps/bin/lmp

# Entrypoint that activates conda env
RUN echo '#!/bin/bash\n\
. /opt/miniconda/etc/profile.d/conda.sh\n\
conda activate env\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
