#!/bin/bash

virtualenv --system-site-packages  --prompt='(pipeline)' ./venv
source ./venv/bin/activate
pip install -r requirements.txt