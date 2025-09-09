import click
import numpy as np
import torch

from stephanie.agents.filter_bank_agent import FilterBankAgent
from stephanie.db import Session
from stephanie.models.skill_filter import SkillFilterORM


@click.group()
def skills(): ...

@skills.command("list")
@click.option("--casebook", default=None, help="Filter by casebook name")
def list_skills(casebook):
    s = Session()
    fba = FilterBankAgent(s)
    for sf in fba.list_filters(casebook):
        click.echo(f"{sf.id}  domain={sf.domain}  align={sf.alignment_score}")

@skills.command("apply-weight")
@click.argument("skill_id")
@click.argument("base_sd_path")
@click.argument("out_sd_path")
@click.option("--alpha", default=1.0, type=float)
def apply_weight(skill_id, base_sd_path, out_sd_path, alpha):
    s = Session(); fba = FilterBankAgent(s)
    sf = fba.load_filter(skill_id)
    sd = torch.load(base_sd_path, map_location="cpu")
    new_sd = fba.apply_weight_filter(sd, sf, alpha=alpha)
    torch.save(new_sd, out_sd_path)
    click.echo(f"Wrote enhanced state_dict to {out_sd_path}")

@skills.command("compose-visual")
@click.argument("skill_ids")
@click.argument("out_npy")
@click.option("--alphas", default=None, help="Comma-separated alphas")
def compose_visual(skill_ids, out_npy, alphas):
    s = Session(); fba = FilterBankAgent(s)
    ids = [x.strip() for x in skill_ids.split(",")]
    fs = [fba.load_filter(i) for i in ids]
    a = [float(x) for x in alphas.split(",")] if alphas else None
    combo = fba.compose_filters(fs, mode="visual", alphas=a)
    np.save(out_npy, combo)
    click.echo(f"Saved composed residual VPM to {out_npy}")
