import os
from dotenv import load_dotenv
from registry import ToolRegistry

load_dotenv()
db_url = os.getenv("DATABASE_URL", "sqlite:///toolregistry.db")
print(f"Initializing registry for {db_url}...")
registry = ToolRegistry(db_url)
print("Registry initialized. Schema should be updated.")
