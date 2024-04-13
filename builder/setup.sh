#!/bin/bash

# NOTE: This script is not ran by default for the template docker image.
#       If you use a custom base image you can add your required system dependencies here.

set -e # Stop script on error
apt-get update && apt-get upgrade -y # Update System

# Install System Dependencies
# - openssh-server: for ssh access and web terminal
apt-get install -y --no-install-recommends software-properties-common curl git openssh-server

# Install Python 3.11
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
apt-get install build-essential cmake libboost-system-dev libboost-thread-dev \
        libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev \
        liblzma-dev -y

# Install pip for Python 3.11
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# Clean up, remove unnecessary packages and help reduce image size
apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

gcloud compute scp /Users/patrickcmd/Projects/sunbirdai/sunbird-asr-stt-transcribe-inference.zip sb-asr-stt-inference-instance:~ --zone "us-west1-b" --project "sb-gcp-project-01"
gcloud compute ssh --zone "us-west1-b" "sb-asr-stt-inference-instance" --project "sb-gcp-project-01"
