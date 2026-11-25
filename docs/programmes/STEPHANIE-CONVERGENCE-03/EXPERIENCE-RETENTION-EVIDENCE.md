# Experience Retention Evidence

Stephanie used the existing ExperimentStore and CaseBookStore through an isolated SQLite database below the experiment root. The retained entry includes observation, hypothesis, candidate and protocol references, Worker execution and evaluator/Praxis references, recommendation, limitations, and decision status. It stores bounded references, not copies of Worker or Praxis databases.

The CaseBook retains the C1 result and unsuccessful/inconclusive C2, C3, and C4 cases; reuse is marked ineligible for automatic action and no candidate was promoted. A subsequent CaseBook retrieval found the prior unsuccessful candidate. This demonstrates retrieval only; it does not show that retrieval improved a later outcome.

Artifacts: `experiences.json`, `retained-experience-retrieval.json`, and `prior-unsuccessful-experience-retrieval.json`.
