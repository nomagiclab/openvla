FROM pzal1/robot_interface_system_deps

RUN git clone https://github.com/nomagiclab/robot-interface.git /workspace/robot-interface
RUN cd /workspace/robot-interface && git checkout 32-rename-everything-according-to-the-new-robot-interface-name

# Install system dependencies for flash-attn.
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    cuda-nvcc-12-1 \
    cuda-cudart-dev-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.1
# Ensure the venv and CUDA bin dirs are in the PATH
ENV PATH="/.venv/bin:${CUDA_HOME}/bin:$PATH"

WORKDIR /workspace
RUN git clone -b 4-investigate-and-likely-switch-to-openvla-oft https://github.com/nomagiclab/openvla.git && \
    cd openvla && \
    git submodule init third_party/lerobot && \
    git submodule update --recursive --init third_party/lerobot 

WORKDIR /workspace/openvla

RUN ls -la third_party/lerobot 

# Ensure pip is functional and clear cache in a separate step
RUN /.venv/bin/python -m ensurepip --upgrade && \
    /.venv/bin/python -m pip cache purge

# Editable install of openvla, then lerobot submodule, then reinstall newly
# missing openvla dependencies to negotiate dependency incompatibility.
# Then install flash-attn separately (per OpenVLA instructions)
# and download the openvla-7b model.
RUN /.venv/bin/python -m pip install -e . && \
    cd third_party/lerobot/ && \
    /.venv/bin/python -m pip install -e . && \
    cd ../../ && \
    MISSING_DEPS=$(/.venv/bin/python -m pip check | awk '$1 ~ /openvla/ {gsub(/,/,"",$5); print $5}' || true) && \
    if [ -n "$MISSING_DEPS" ]; then /.venv/bin/python -m pip install $MISSING_DEPS; fi && \
    /.venv/bin/python -m pip install packaging ninja && \
    /.venv/bin/python -m pip install "flash-attn==2.5.5" --no-build-isolation && \
    /.venv/bin/python -m pip install huggingface-hub && \
    huggingface-cli download openvla/openvla-7b
