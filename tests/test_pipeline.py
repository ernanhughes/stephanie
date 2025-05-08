from co_ai.supervisor import Supervisor

def test_supervisor_pipeline_smoke():
    supervisor = Supervisor()
    assert supervisor is not None
