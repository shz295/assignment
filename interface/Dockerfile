FROM python:3.11

WORKDIR /interface

COPY requirements.txt /interface

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /interface

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "interface.py", "--", "--api_address", "api"]