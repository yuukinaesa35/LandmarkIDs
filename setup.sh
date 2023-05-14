#!/bin/bash

# Install required system libraries
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
sudo apt-get install -y libgl1-mesa-glx

# Install required Python packages
pip install -r requirements.txt

mkdir -p ~/.streamlit/


echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
