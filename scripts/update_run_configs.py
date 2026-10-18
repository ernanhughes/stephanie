import argparse
import json

from sqlalchemy.orm import sessionmaker

from stephanie.orm import PipelineRunORM
from stephanie.orm.base import create_database_engine_from_hydra


def update_run_configs(session, *, apply: bool = False) -> int:
    fix_count = 0
    for run in session.query(PipelineRunORM).all():
        updated = False
        if isinstance(run.run_config, str):
            try:
                run.run_config = json.loads(run.run_config)
                updated = True
            except json.JSONDecodeError:
                print(f"Invalid JSON in run_config for run_id {run.run_id}")

        if isinstance(run.pipeline, str):
            try:
                run.pipeline = json.loads(run.pipeline)
                updated = True
            except json.JSONDecodeError:
                print(f"Invalid JSON in pipeline for run_id {run.run_id}")

        fix_count += int(updated)

    if apply:
        session.commit()
        print(f"Applied pipeline_run updates to {fix_count} rows.")
    else:
        session.rollback()
        print(f"Dry run: {fix_count} pipeline_run rows would be updated; pass --apply to write.")
    return fix_count


def main():
    parser = argparse.ArgumentParser(description="Normalize legacy pipeline run JSON fields")
    parser.add_argument("--apply", action="store_true", help="Commit updates (default: dry run)")
    args = parser.parse_args()

    engine = create_database_engine_from_hydra("config")
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        update_run_configs(session, apply=args.apply)
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
