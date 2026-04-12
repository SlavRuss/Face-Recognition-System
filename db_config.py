import os

DB_HOST = os.getenv("DB_HOST", "**********")
DB_USER = os.getenv("DB_USER", "**********")
DB_PASSWORD = os.getenv("DB_PASSWORD", "*********")
DB_NAME = os.getenv("DB_NAME", "*********")
DB_SCHEMA = os.getenv("DB_SCHEMA", "*************")

DB_CONFIG = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "dbname": DB_NAME,
    "options": f"-c search_path={DB_SCHEMA}",
}