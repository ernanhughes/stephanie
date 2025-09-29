from __future__ import annotations

import json
from typing import Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL

KABOOK = "[KA] Knowledge Atoms"

def _nearest_windows(memory, query_text: str, k=2):
    q = memory.embeddings.get_or_create(query_text or "")
    hits = memory.embeddings.find_neighbors(q, k=20)
    return [h.get("text","") for h in hits if "[WIN:" in (h.get("text","") or "")][:k]

class KAMinerAgent(BaseAgent):
    """
    Mines high-K turns into structured Knowledge Atoms and writes them to a CaseBook.
    Requires a trained KRM in container under key 'krm' and an LLM under 'llm'.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.krm = container.get("krm")
        self.llm = container.get("llm")
        self.threshold = float(cfg.get("threshold", 0.70))
        self.max_convs = int(cfg.get("max_conversations", 500))

    async def run(self, context: Dict) -> Dict:
        goal = context.get(GOAL, {})
        self.report({"event":"start", "threshold": self.threshold})

        cb = self.memory.casebooks.ensure_casebook(
            name=KABOOK,
            pipeline_run_id=context.get("pipeline_run_id"),
            description="Applied Knowledge Atoms mined from high-K turns."
        )

        top = self.memory.chats.get_top_conversations(limit=self.max_convs, by="turns")
        created = 0

        for conv, _ in top:
            turns = self.memory.chats.get_turns_for_conversation(conv.id)
            msgs  = self.memory.chats.get_messages(conv.id)
            msg_by_id = {m.id: m for m in msgs}

            for t in turns:
                if not t.user_message or not t.assistant_message:
                    continue
                u = t.user_message.text or ""; a = t.assistant_message.text or ""
                if not u.strip() or not a.strip():
                    continue

                # next user
                next_user_text = ""
                a_msg = msg_by_id.get(t.assistant_message_id)
                if a_msg:
                    next_user = next((m for m in msgs if m.order_index > a_msg.order_index and m.role=="user"), None)
                    next_user_text = (next_user.text if next_user else "") or ""

                wins = _nearest_windows(self.memory, u, k=2)
                score = self.krm.score_turn(f"USER: {u}\nASSISTANT: {a}", next_user_text, wins)

                if score < self.threshold:
                    continue

                prompt = f"""
You are a knowledge miner. Convert the assistant's answer into a structured Knowledge Atom (KA).
KA types: "decision", "rule", "recipe", "pattern", "checklist".

[USER]
{u}

[ASSISTANT]
{a}

[RETRIEVED_WINDOWS]
{chr(10).join(w[:1000] for w in wins)}

Return strict JSON with keys:
type, statement, conditions (string[]), steps (string[]), rationale, inputs (string[]), outputs (string[]),
citations (string[]: include any [WIN:x:y] or [TURN:{t.id}] if you relied on them),
applicability_tags (string[] from: ["self_improving_systems","q_learning","embeddings","hrm","memento","cbr","casebook_reasoning","pacs","ner_retriever"]).
JSON:
"""
                try:
                    raw = self.llm.generate(prompt, max_tokens=500)  # use your bridge
                    ka = json.loads(raw)
                except Exception as e:
                    self.report({"event":"llm_parse_error","turn_id":t.id,"error":str(e)})
                    continue

                # finalize + save
                ka_id = f"KA-{t.id}"
                ka["id"] = ka_id
                ka["confidence"] = round(float(score), 3)
                ka["origin"] = {"conversation_id": conv.id, "turn_id": t.id}

                case = self.memory.casebooks.add_case(
                    casebook_id=cb.id,
                    goal_text="Applied Knowledge",
                    agent_name="ka_miner",
                    prompt_text=ka.get("statement",""),
                    scorables=[{
                        "text": json.dumps(ka, ensure_ascii=False),
                        "role": "assistant",
                        "source": "KA",
                        "meta": {"ka_id": ka_id, "conv_id": conv.id, "turn_id": t.id}
                    }]
                )
                # index KA text for retrieval
                self.memory.embeddings.get_or_create(f"[KA:{ka_id}] {ka.get('statement','')}")
                created += 1
                self.report({"event":"ka_created","ka_id":ka_id,"case_id":case.id,"score":score})

        self.report({"event":"done","created":created})
        context["ka_created"] = created
        return context
