#!/bin/bash

# Docker entrypoint. this script runs when you run the docker image. 

# Don't forget to map the port 8501 to your local machine, otherwise you won't be able to access the app.


(cd /background_switcher/ && exec streamlit run app.py)