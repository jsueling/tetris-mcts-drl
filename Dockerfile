# GPU supported image
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    vim \
    # Create a symlink between python3 and python
    && ln -s /usr/bin/python3 /usr/bin/python \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip3 install --no-cache-dir poetry

# Copy project files
COPY . /app/

# Install Python dependencies without installing the current project
RUN poetry install

# Opens a shell in the container
CMD ["/bin/bash"]
