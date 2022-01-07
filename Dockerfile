FROM tensorflow/tensorflow:latest
RUN apt-get update && \
  apt-get -y upgrade
# Install requirements 
# Copy everything over 
COPY . .
