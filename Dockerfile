FROM intel/intel-optimized-tensorflow:latest
WORKDIR /root
COPY script.py /root/
COPY myh5model.h5 /root/

RUN pip3 install pandas matplotlib numpy datetime python-binance
