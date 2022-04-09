FROM amancevice/pandas:latest
WORKDIR /root
COPY script.py /root/
COPY myh5model.h5 /root/
COPY requirements-my.txt /root/

RUN apt update && apt upgrade -y && pip3 install -r requirements-my.txt && apt install -y pip install python-binance

