import json
from sqlalchemy.orm import sessionmaker
from stephanie.models import PipelineRunORM
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .envs
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

engine = create_engine("postgresql://co:co@localhost/co")
Session = sessionmaker(bind=engine)
session = Session()



Session = sessionmaker(bind=engine)
session = Session()

fix_count = 0

for run in session.query(PipelineRunORM).all():
    updated = False

    # Fix run_config
    if isinstance(run.run_config, str):
        try:
            run.run_config = json.loads(run.run_config)
            updated = True
        except json.JSONDecodeError:
            print(f"Invalid JSON in run_config for run_id {run.run_id}")

    # Fix pipeline
    if isinstance(run.pipeline, str):
        try:
            run.pipeline = json.loads(run.pipeline)
            updated = True
        except json.JSONDecodeError:
            print(f"Invalid JSON in pipeline for run_id {run.run_id}")

    if updated:
        fix_count += 1

print(f"✅ Fixed {fix_count} pipeline_run entries.")

session.commit()
session.close()
