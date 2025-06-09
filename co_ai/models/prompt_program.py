import uuid

from sqlalchemy import JSON, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class PromptProgramORM(Base):
    __tablename__ = "prompt_programs"

    id = Column(String, primary_key=True, default=generate_uuid)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id", ondelete="SET NULL"), nullable=True)
    goal = Column(Text, nullable=False)
    template = Column(Text, nullable=False)
    inputs = Column(JSON, default={})
    version = Column(Integer, default=1)
    parent_id = Column(String, ForeignKey("prompt_programs.id"), nullable=True)
    strategy = Column(String, default="default")
    prompt_text = Column(Text, nullable=True)
    hypothesis = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)
    mutation_type = Column(String, nullable=True)
    execution_trace = Column(Text, nullable=True)
    extra_data = Column(JSON, default={})

 
    parent = relationship("PromptProgramORM", remote_side=[id], backref="children")
    prompt = relationship("PromptORM", backref="prompt_programs")
    pipeline_run = relationship("PipelineRunORM", back_populates="prompt_programs")

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "template": self.template,
            "inputs": self.inputs,
            "version": self.version,
            "parent_id": self.parent_id,
            "prompt_id": self.prompt_id,
            "propipeline_run_idmpt_id": self.pipeline_run_id,
            "strategy": self.strategy,
            "prompt_text": self.prompt_text,
            "hypothesis": self.hypothesis,
            "score": self.score,
            "rationale": self.rationale,
            "mutation_type": self.mutation_type,
            "execution_trace": self.execution_trace,
            "extra_data": self.extra_data,
        }
