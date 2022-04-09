FROM amancevice/pandas:latest
WORKDIR /root
COPY script.py /root/
COPY myh5model.h5 /root/
COPY requirements-my.txt /root/

RUN pip3 install tensorflow pandas matplotlib numpy datetime
#RUN apt update && apt upgrade -y && pip3 install -r requirements-my.txt && pip3 install python-binance

