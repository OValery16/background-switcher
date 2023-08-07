# vim: ft=dockerfile
# From Dockerfil-gpu-base
FROM ubuntu

USER 0

ENV DEBIAN_FRONTEND=noninteractive
ENV AWS_DEFAULT_REGION="eu-west-1"
ENV PYTHONUNBUFFERED 1
ENV TZ=Europe/Paris

WORKDIR /background_switcher

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

COPY ./requirements.txt /background_switcher

RUN pip3 install --no-cache-dir -r /background_switcher/requirements.txt

COPY . /background_switcher/

CMD ["bash"]

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]