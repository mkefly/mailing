# Start with a Linux Base Image
FROM python:3.12.0-slim-bookworm AS builder

# Install Java
RUN apt update && \
    apt install -y sudo && \
    sudo apt install default-jdk -y

RUN sudo apt install git-all -y

# Install OS packages
RUN apt-get update && apt-get -y upgrade && apt-get -y install apt-utils

# Copy install scripts
ENV REPOSITORY_NAME "data-stream"
WORKDIR /workspaces/${REPOSITORY_NAME}
COPY . /workspaces/${REPOSITORY_NAME}/.

# # Install the Project
RUN pip install -r requirements.txt

CMD ["sleep", "infinity"]
