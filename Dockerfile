FROM python:3.9

WORKDIR /dangqu.pt.ai

COPY requirements.txt .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD gunicorn -c gunicorn_config.py main:app