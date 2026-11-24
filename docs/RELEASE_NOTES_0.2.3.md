## Onset 0.2.3 for Mac

A cleaner Onset, and answers about your own Mac. Onset 0.2.0 and 0.2.1 update themselves to this version (or Onset >
Check for Updates…).

**Download:** `Onset-0.2.3-arm64.dmg` below. Signed with Whissle LLC's Developer ID and notarized by Apple. Check
the download: `shasum -a 256 Onset-0.2.3-arm64.dmg` should match `Onset-0.2.3-arm64.dmg.sha256` (`ac24e74dd95bfcbf9baf6970290360484d6fa4f28190bfb04f68da3d8cbe1052`).

### New
- **One clean screen.** Your question, the answer and where it came from. Hold Space or click the mic to talk, or
  type. The side-by-side comparison, timeline and question list are gone from the app.
- **Your Mac.** "Find my resume" lists matching files (click one to open it). "Play some jazz", "pause", "next song"
  control Music. "Open Safari" opens an app. macOS asks once before Onset controls Music or reads a folder.
- **No setup screen.** Onset opens straight to asking: answers are written on your Mac and the web search is free
  (DuckDuckGo). Your own service keys are optional, under Settings > Advanced.
- **When the on-device answer can't do the question justice** (lists, recommendations), Onset shows the top results
  instead of a page title.

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
- The on-device answer model is small: for open questions you get the top results rather than a written list.
- Finding files searches file names when Spotlight has not indexed your Mac.
- English only.
