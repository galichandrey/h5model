FROM amancevice/pandas:latest
RUN apt-get update &&
apt-get upgrade -y &&
pip3 install -r requirements-dev.txt &&
apt-get install -y pip install python-binance
WORKDIR /root
COPY script.py /root/
COPY myh5model.h5 /root/
