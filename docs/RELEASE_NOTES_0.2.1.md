## Onset 0.2.1 for Mac

A fix release for answers in a conversation. Onset 0.2.0 updates itself to this version (or Onset > Check for
Updates…).

**Download:** `Onset-0.2.1-arm64.dmg` below. Signed with Whissle LLC's Developer ID and notarized by Apple. Check
the download: `shasum -a 256 Onset-0.2.1-arm64.dmg` should match `Onset-0.2.1-arm64.dmg.sha256` (`cba93eb223afa4ba4f7a1417a50598b266ff47048a5db165c9d8d334716ce1eb`).

### Fixed
- **A new question could get the last question's answer.** Asking "what are five good books to read" right after
  "who is the prime minister of India" could answer with sources about prime ministers. Each question now answers
  from its own search; the last question's sources are used only for follow-ups like "how old is he?".
- **"Who is the president of India" said the prime minister.** Onset now looks up a country's president and prime
  minister separately, and states facts like these exactly as Wikidata has them.
- **Ages were guessed.** "How old is…" is now worked out from the birth date.
- **Wrong answers could be remembered.** Onset no longer saves its on-device answers as checked facts. Answers saved
  by 0.2.0 are set aside on first launch (kept in `~/Library/Application Support/Onset/data/keystone.v1`).
- **Follow-ups like "and what about Canada?"** now carry the topic of the previous question.

### Known limits
- The on-device answer model is small. For open questions ("five good books to read") it can name things the
  sources don't support; check the cited sources, or add an Anthropic key in Settings for Claude-written answers.
- English only.
