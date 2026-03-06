import psycopg2

from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("DATABASE_URL"))
conn = psycopg2.connect("postgresql://langgraph:langgraph@127.0.0.1:5432/toolregistry")
cur = conn.cursor()
cur.execute("SELECT NOW();")
print(cur.fetchone())
conn.close()
