FROM python:3.11
# !!!
WORKDIR /practice-cnn
# !!!
COPY requirements.txt .
COPY myapp/ /practice-cnn/myapp/
COPY static/ /practice-cnn/static/
COPY emnist-balanced-mapping.txt .

# !!!
#COPY requirements.txt /workdir/
#COPY myapp/ /workdir/myapp/
#COPY static/ /workdir/static/
#COPY emnist-balanced-mapping.txt /workdir/

# !!!
#WORKDIR /workdir

# !!!
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python

# Run the application
CMD ["uvicorn", "myapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
