FROM python:3.11.5
RUN pip install --upgrade pip
COPY ./app/requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY ./app /app
EXPOSE 5000
COPY ./entrypoints.sh /entrypoints.sh
RUN chmod +x /entrypoints.sh
ENTRYPOINT ["/entrypoints.sh"]