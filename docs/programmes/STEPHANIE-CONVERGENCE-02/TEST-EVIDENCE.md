# Test evidence

Focused runs (Python 3.13):

- AI `tests/inference/test_inference.py` selected structured success/failure/trace cases: 10 passed, foreground 1.1 s; individual calls under 0.005 s. A prior broader invocation ran 26 nodes with one failed assertion about fake-provider usage accounting (`estimated`, not `unknown`); assertion was corrected to accept the provider's explicit estimate. No behavior defect was found.
- Stephanie `tests/test_sc2_integrations.py`: 6 passed in 1.32 s; slowest test 0.54 s. It covers loopback-only GETs, source identities/UNKNOWN states, Praxis callable insufficiency, local investigation/proposal persistence, CaseBook retention policy, unavailable Worker, and malformed AI citation evidence preservation.
- Stephanie integration plus existing improvement tests: 12 passed in 15.57 s; one legacy local SQLite persistence test took 14.00 s. This >2 s check is deliberately not part of the routine stage loop going forward and should be marked long under repository policy.

Long suites, full backend/frontend suites and operational test routes were not run. An earlier focused Stephanie invocation unintentionally included a 14 s SQLite persistence test; it is now marked `@pytest.mark.long` and was not part of the final focused loop. AI runner emitted a PytestRemovedIn10Warning from installed AnyIO.
