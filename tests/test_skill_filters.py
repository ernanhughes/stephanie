import numpy as np
import torch
from collections import OrderedDict
from stephanie.agents.filter_bank import FilterBankAgent
from stephanie.models.skill_filter import SkillFilterORM

def test_visual_apply_safety(tmp_path, session):
    # Fake residual
    res = (np.random.rand(10, 3).astype(np.float32) * 0.1)
    res_path = tmp_path / "res.npy"
    np.save(res_path, res)

    sf = SkillFilterORM(id="test", casebook_id="cb1", domain="general",
                        vpm_residual_path=str(res_path), weight_delta_path=None)
    session.add(sf); 
    session.commit()

    fba = FilterBankAgent(session)
    base = np.zeros((10,3), dtype=np.float32)
    out = fba.apply_visual_filter(base, sf, alpha=1.0)
    assert out.shape == base.shape
    assert np.all(out >= 0) and np.all(out <= 1)

def test_weight_apply_safety(tmp_path, session):
    # Fake weight delta
    v = OrderedDict([("lm.head.weight", torch.zeros(5,5))])
    wp = tmp_path / "w.pt"
    torch.save(v, wp)

    sf = SkillFilterORM(id="test2", casebook_id="cb1", domain="general",
                        weight_delta_path=str(wp))
    session.add(sf); session.commit()

    fba = FilterBankAgent(session)
    model_sd = OrderedDict([("lm.head.weight", torch.zeros(5,5))])
    new_sd = fba.apply_weight_filter(model_sd, sf, alpha=1.0)
    assert "lm.head.weight" in new_sd
