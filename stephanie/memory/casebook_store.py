# stephanie/memory/casebook_store.py
from sqlalchemy.orm import Session
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM

class CaseBookStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "casebooks"

    def create_casebook(self, name, description=""):
        cb = CaseBookORM(name=name, description=description)
        self.session.add(cb)
        self.session.commit()
        return cb

    def add_case(self, casebook_id, goal_id, goal_text, agent_name,
                 mars_summary=None, scores=None, metadata=None,
                 scorables=None):
        case = CaseORM(
            casebook_id=casebook_id,
            goal_id=goal_id,
            goal_text=goal_text,
            agent_name=agent_name,
            mars_summary=mars_summary,
            scores=scores,
            meta=metadata,
        )
        self.session.add(case)
        self.session.flush()

        if scorables:
            for s in scorables:
                cs = CaseScorableORM(
                    case_id=case.id,
                    scorable_id=s["id"],
                    scorable_type=s.get("type"),
                    role=s.get("role", "input"),
                )
                self.session.add(cs)

        self.session.commit()
        return case

    def get_cases_for_goal(self, goal_id):
        return self.session.query(CaseORM).filter_by(goal_id=goal_id).all()

    def get_cases_for_agent(self, agent_name):
        return self.session.query(CaseORM).filter_by(agent_name=agent_name).all()
