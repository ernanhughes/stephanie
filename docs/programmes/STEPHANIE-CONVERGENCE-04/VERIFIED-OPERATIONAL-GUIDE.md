# Verified owner guide — isolated SC4B review

The completed owner-review experiment remains in the separate local review workspace:

1. Open [Mission Improvement — isolated review](http://127.0.0.1:8791/improvement). The process is separate from the normal Mission app on `:8790`.
2. Locate proposal `wp:1ee64e8049444480`, version 2, and the linked investigation/experiment. The registered envelope is `we:08f10a4a732848c5`.
3. Inspect the frozen candidate `034647ef6575a8a58461a4c8`, digest `b08613481dbe20e50a6f2e14bc850a0bd74581852a31297b075a5104805d273d`, and protocol digest `2373dd5ee2262af66b7a260da6ad27df62a1ce9f5a0c62d8e9be14016cd401c6`.
4. Inspect the Worker execution ledger and saved C1–C7 observations, then the Praxis C1–C5 comparison and the retained experience records. The experiment ID is `7cc37cb2e95dd58d098c3d80`.
5. `reported_complete` means the bounded investigation/experiment completed. It does not mean the candidate was accepted. The formal post-result decision was left undecided. Use the page's result decision control only after your own review; accept/reject is your decision. Neither outcome authorizes production promotion.

On 2026-11-25, read-only checks of the isolated proposal and investigation detail confirmed the proposal state, experiment/candidate/protocol identities, four retained records, and zero recorded human decisions. `GET /api/improvement/records` is a separate artifact feed and does not contain this owner-review experiment.

The URL and current process were read-only checked on 2026-11-25. The normal `:8790` runtime reports a different isolated artifact (`4205bddf27071a214ba99e45`); do not confuse it with this owner-review record or dispatch a new experiment from the normal app. If the `:8791` page is unavailable, stop there rather than switching to `:8790` for dispatch.
