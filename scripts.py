import redis
import os
import certifi
import ssl
import dotenv

dotenv.load_dotenv()

r = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6380)),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=True,
    ssl_cert_reqs=ssl.CERT_REQUIRED,
    ssl_ca_certs=certifi.where(),
    decode_responses=True
)

print(r.ping())