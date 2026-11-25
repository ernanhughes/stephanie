# Verified user guide — SC4 source state and operational status

The Improvement workspace source is at Mission `/improvement`. Its source controls now support:

1. Select a Mission episode and open finding.
2. Create a Work Corrective investigation request and approve it through the existing Mission workflow.
3. Request and inspect Stephanie's bounded investigation.
4. For the specifically supported Worker JSON-fence profile only, select a current Mission objective and request a separate experiment decision. The same Work Corrective proposal is revised, versioned, and returned to draft for renewed approval.
5. After approval, register a matching active Work Envelope and dispatch the frozen experiment once.
6. Review saved baseline/candidate evaluation, Praxis comparisons, resources and retained experience. Execution completion is separate from objective completion, acceptance and promotion.

The page exposes actions in source, and focused frontend component/API tests plus the frontend production build pass. A browser end-to-end with a current Mission backend has not been performed.

The normal service at `http://127.0.0.1:8790` remains the pre-SC4 instance. Read-only verification reports version `0.1.0`, schema 54, and `GET /api/improvement/integration` returns 404. The new controls are therefore **not operationally active**. Do not use the current running instance for investigation or dispatch; activation requires the remaining SC4B and SC4C gates.
