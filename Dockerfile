FROM python:3.6.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY clustinator ./

RUN wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh; chmod +x wait-for-it.sh

ENTRYPOINT ./wait-for-it.sh -t 120 ${RABBITMQ:-rabbitmq}:5672 -- python -u ./receiver.py --rabbitmq=${RABBITMQ:-rabbitmq}