# Oracle protection

E1's failed call path was: fixture oracle-labelled raw row → `Evidence` schema accepted the field → import persisted it → `investigate_with_ai()` serialized complete models → oracle metadata entered prompt. The retained call fingerprint was `a0b8bb230c956111`; it remains ineligible and is preserved.

SE2 now rejects protected key names, hidden/development visibility, contaminated integrity, and current experiment outcomes before generation. AI evidence serialization is field-whitelisted. Experience use requires explicit historical owner review, model visibility, trusted sanitized integrity, and another protected-field scan. Current evaluation rubric is kept out of generation payload construction. Tests cover oracle labels, hidden answers, current outcomes, development-only content, contaminated data, and legitimate reviewed history.

These are structural guards, not semantic DLP: disguised sensitive prose cannot be reliably detected by key scanning alone. Human review and sanitized source construction remain required.
