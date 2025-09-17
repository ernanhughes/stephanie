# stephanie/knowledge/casebook_store.py
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CaseBook:
    id: str
    name: str
    tags: List[str]
    meta: Dict[str, Any]

@dataclass
class Case:
    id: str
    casebook_id: str
    prompt_text: str
    agent_name: str
    meta: Dict[str, Any]

@dataclass
class Scorable:
    id: str
    case_id: str
    role: str         # "code", "text", "tests", "vpm", "dpo_pair", "edit_snippet"
    text: str
    scorable_type: str
    meta: Dict[str, Any]

class CaseBookStore:
    def __init__(self, root: str = "./runs/knowledge/casebooks"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _book_path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def ensure_casebook(self, name: str, tags: List[str], meta: Dict[str, Any]) -> CaseBook:
        p = self._book_path(name)
        if p.exists():
            d = json.loads(p.read_text())
            return CaseBook(**d["casebook"])
        cb = CaseBook(id=uuid.uuid4().hex, name=name, tags=tags, meta=meta)
        p.write_text(json.dumps({"casebook": asdict(cb), "cases": [], "scorables": []}, indent=2))
        return cb

    def add_case(self, casebook_name: str, prompt_text: str, agent_name: str, meta: Dict[str, Any]) -> Case:
        p = self._book_path(casebook_name); d = json.loads(p.read_text())
        c = Case(id=uuid.uuid4().hex, casebook_id=d["casebook"]["id"], prompt_text=prompt_text, agent_name=agent_name, meta=meta)
        d["cases"].append(asdict(c)); p.write_text(json.dumps(d, indent=2)); return c

    def add_scorable(self, casebook_name: str, case_id: str, role: str, text: str, scorable_type: str, meta: Dict[str, Any]) -> Scorable:
        p = self._book_path(casebook_name); d = json.loads(p.read_text())
        s = Scorable(id=uuid.uuid4().hex, case_id=case_id, role=role, text=text, scorable_type=scorable_type, meta=meta)
        d["scorables"].append(asdict(s)); p.write_text(json.dumps(d, indent=2)); return s

    def list_casebooks_by_tag(self, tag: str) -> List[CaseBook]:
        out = []
        for fp in self.root.glob("*.json"):
            d = json.loads(fp.read_text())
            if tag in d["casebook"]["tags"]:
                out.append(CaseBook(**d["casebook"]))
        return out

    def load(self, name: str) -> Dict[str, Any]:
        return json.loads(self._book_path(name).read_text())
