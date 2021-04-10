FROM python:3.8-slim
WORKDIR /app
COPY ./pdemo /app/pdemo
COPY requirements.txt /app
COPY setup.py /app
COPY cardinal_data_capture.mp4 /app
COPY app.py /app
RUN cd /app && pip install .
CMD streamlit run app.py
