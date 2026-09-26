ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG USERNAME=nlee
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git for VS Code, libgl1/libglib2.0-0 so opencv-python can import
RUN apt-get update && apt-get install -y --no-install-recommends \
        git sudo libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# Python packages (own layer, so code changes don't trigger a reinstall)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

USER ${USERNAME}