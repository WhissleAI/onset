## Onset 0.2.4 for Mac

Asking out loud is reliable again. Onset 0.2.x updates itself to this version (or Onset > Check for Updates…).

**Download:** `Onset-0.2.4-arm64.dmg` below. Signed with Whissle LLC's Developer ID and notarized by Apple. Check
the download: `shasum -a 256 Onset-0.2.4-arm64.dmg` should match `Onset-0.2.4-arm64.dmg.sha256` (`SHA_PENDING`).

### Fixed
- **Spoken questions and follow-ups that showed but got no answer.** Clicking the mic now means "I'm asking you":
  Onset answers what you say, then stops listening. (It used to keep listening and skip anything it wasn't sure was
  meant for it, which caught real questions and follow-ups.) It stops by itself if it hears nothing for 8 seconds.
- **If Onset ever holds something back,** it now shows what it heard with an "Answer it" button instead of nothing.
- **"Find my resume" found nothing** when Spotlight was slow. Onset now falls back to searching your Desktop,
  Documents and Downloads, puts documents (PDFs, Word, Pages) ahead of work files, and lists each file once.

### Known limits
- The on-device answer model is small: for open questions you get the top results rather than a written list.
- English only.
