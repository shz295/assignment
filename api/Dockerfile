FROM borda/docker_python-opencv-ffmpeg:cpu-py3.11-cv4.8.0

RUN mkdir /api

COPY requirements.txt /api

WORKDIR /api

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]