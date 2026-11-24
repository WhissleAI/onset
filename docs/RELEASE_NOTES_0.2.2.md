## Onset 0.2.2 for Mac

Follow-up questions. Onset 0.2.1 updates itself to this version (or Onset > Check for Updates…).

**Download:** `Onset-0.2.2-arm64.dmg` below. Signed with Whissle LLC's Developer ID and notarized by Apple. Check
the download: `shasum -a 256 Onset-0.2.2-arm64.dmg` should match `Onset-0.2.2-arm64.dmg.sha256` (`SHA_PENDING`).

### Fixed
- **Short follow-ups found no sources.** "Who is the prime minister of India", then "and USA", answered "I couldn't
  find sources": Onset threw away the right source because it says "United States", not "USA". Common short names
  (USA, UK, UAE, NYC, …) now match their full names, and "UK" no longer looks up Uttarakhand.
- **A follow-up could repeat the last answer.** "And USA" could say India's prime minister again. A fact answer now
  has to be about the place or person you just asked about.
- **"Prime minister" of a country that has none.** Onset now says so: "The United States has no prime minister; its
  head of government is the president, Donald Trump."
- **Fact lookups dropped out for several questions after a slow one.** A slow connection to Wikidata now pauses it for
  a few seconds, not twenty, so the next questions still get checked facts.
- **"Five good books to read" was treated as a fact lookup.** List and recommendation questions now go to the web.
- **Confidence.** An answer with no sources, or a very short answer the sources don't confirm, no longer shows
  "High confidence".

### Known limits
- The on-device answer model is small. For open questions ("five good books to read") it often quotes a source line
  instead of writing a list; add an Anthropic key in Settings for Claude-written answers.
- English only.
