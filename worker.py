"""Define worker instance for heroku dyno."""
import os

import redis
from rq import Worker, Queue, Connection
import urllib.parse as up

listen = ["high", "default", "low"]

redis_url = os.getenv("REDISTOGO_URL")

up.uses_netloc.append("redis")
url = up.urlparse(redis_url)
# conn = Redis(host=url.hostname, port=url.port, db=0, password=url.password)

conn = redis.from_url(redis_url)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()
