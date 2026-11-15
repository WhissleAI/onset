import AppKit
import WebKit

/// Unattended checks for the build (MACAPP.md "Verify"): launch with
///   open "dist/<APP>.app" --args -snapshotDir /path -autoReplay q01.wav [-autoQuit 1]
/// and the app saves PNG snapshots of its own window (splash while warming, the page, the page after the replay)
/// plus the page text after the replay (page_after_replay.txt). Off unless -snapshotDir is given.
final class Automation {
    static let shared = Automation()
    private let dir: URL? = UserDefaults.standard.string(forKey: "snapshotDir").map { URL(fileURLWithPath: $0, isDirectory: true) }
    private let replayFile = UserDefaults.standard.string(forKey: "autoReplay")
    private let autoQuit = UserDefaults.standard.bool(forKey: "autoQuit")
    /// -autoMic "4,0.3,6,14": push-to-talk holds of these seconds through the page's own mic path (same code as Space),
    /// 1 s apart (the second press starts while the first run is still flushing: the stale run_done race). Each
    /// release logs the page's capture counters; the server logs them too ("stop (client) | page: ...").
    private let autoMic = UserDefaults.standard.string(forKey: "autoMic")
    /// -autoType "q1|q2|...": each question typed into the page's text box one character at a time through the web
    /// view's own text input (insertText: the page sees trusted input events, as from the keyboard), Enter spam on every
    /// second one, a new question while the last is still answering on every third; logs focus, the page's run state,
    /// the WebContent memory (page heap, DOM size) after each, and any page-process end. -autoTypeRounds repeats the list.
    private let autoType = UserDefaults.standard.string(forKey: "autoType")
    private let autoTypeRounds = max(1, UserDefaults.standard.integer(forKey: "autoTypeRounds"))
    /// -autoTypeKeys YES: type with synthesized key events (keyDown / keyUp sent to the web view, Return as key code 36)
    /// instead of insertText + requestSubmit: the whole AppKit key path (text input client, key equivalents).
    private let autoTypeKeys = UserDefaults.standard.bool(forKey: "autoTypeKeys")

    private func key(_ app: AppDelegate, _ chars: String, code: UInt16) {
        for type in [NSEvent.EventType.keyDown, .keyUp] {
            if let e = NSEvent.keyEvent(with: type, location: .zero, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                                        windowNumber: app.window.windowNumber, context: nil, characters: chars,
                                        charactersIgnoringModifiers: chars, isARepeat: false, keyCode: code) {
                if type == .keyDown { app.web.keyDown(with: e) } else { app.web.keyUp(with: e) }
            }
        }
    }
    /// -autoKillPage YES: end the page's web content process once, 4 s after the first page load (test of the app's
    /// recovery: log, reload, plain notice), via WebKit's test hook; snapshot 07_after_page_end.png after the reload.
    private let autoKillPage = UserDefaults.standard.bool(forKey: "autoKillPage")
    private var killed = false
    private var splashShot = false
    private var pageDone = false
    private var log: [String] = []

    func note(_ s: String) {
        guard let d = dir else { return }
        log.append("\(Date().timeIntervalSince1970) \(s)")
        try? log.joined(separator: "\n").write(to: d.appendingPathComponent("automation.log"), atomically: true, encoding: .utf8)
    }

    func stateChanged(_ s: ServerManager.State, app: AppDelegate) {
        guard let d = dir else { return }
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        note("state \(s)")
        if case .starting(let secs, _) = s, secs >= 3, !splashShot {
            splashShot = true
            DispatchQueue.main.async { self.saveView(app.splashHost, "01_splash_warming.png") }
        }
    }

    func pageLoaded(app: AppDelegate) {
        if dir != nil, killed, pageDone {
            note("page reloaded after the page process ended")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                app.web.evaluateJavaScript("(document.getElementById('pageNotice') || {}).textContent || ''") { r, _ in
                    self.note("notice: \((r as? String) ?? "")")
                    self.saveWeb(app.web, "07_after_page_end.png") { self.finish(app) }
                }
            }
            return
        }
        if dir != nil, autoKillPage, !killed {
            killed = true
            pageDone = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                self.note("ending the page process")
                let sel = Selector(("_killWebContentProcess"))
                if app.web.responds(to: sel) { app.web.perform(sel) } else { self.note("no test hook") }
            }
            return
        }
        guard dir != nil, !pageDone else { return }
        pageDone = true
        note("page loaded")
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            self.saveWeb(app.web, "02_ui_loaded.png") {
                if let t = self.autoType {
                    let qs = t.split(separator: "|").map(String.init)
                    let all = Array((0..<self.autoTypeRounds).map { _ in qs }.joined())
                    return self.typeSequence(all, index: 0, app: app)
                }
                if let m = self.autoMic {
                    let holds = m.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
                    return self.micSequence(holds, index: 0, app: app)
                }
                guard let f = self.replayFile else { return self.finish(app) }
                app.web.evaluateJavaScript("window.__hs && window.__hs.ready") { r, _ in self.note("ready=\(String(describing: r))") }
                // mic preconditions (the permission prompt itself needs a human click)
                app.web.evaluateJavaScript("JSON.stringify({secure: window.isSecureContext, getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia), audioWorklet: !!(window.AudioContext && AudioContext.prototype && 'audioWorklet' in AudioContext.prototype)})") { r, _ in
                    self.note("mic preconditions \((r as? String) ?? "?")")
                }
                // -autoReplay a.wav,b.wav,...: each replayed through the page in turn, the next one once the page's run
                // has ended (as a person clicking through the list would); then the snapshots of the LAST one
                let files = f.split(separator: ",").map(String.init)
                self.replaySequence(files, index: 0, app: app) {
                DispatchQueue.main.asyncAfter(deadline: .now() + 4) { self.saveWeb(app.web, "03_replay_speaking.png") {} }
                DispatchQueue.main.asyncAfter(deadline: .now() + 14) {
                    self.saveWeb(app.web, "04_replay_answered.png") {
                        app.web.evaluateJavaScript("document.body.innerText") { r, _ in
                            try? ((r as? String) ?? "").write(to: self.dir!.appendingPathComponent("page_after_replay.txt"),
                                                               atomically: true, encoding: .utf8)
                            self.note("replay menu items: \(app.replayMenu.items.count)")
                            app.toggleDetails()        // View > Show Details: the question list + timeline
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                self.saveWeb(app.web, "05_details_question_list.png") {
                                    app.toggleDetails()
                                    self.finish(app)
                                }
                            }
                        }
                    }
                }
                }
            }
        }
    }

    /// Replays files[0..<n-1] one after another (waiting for each page run to end), then starts the last one and
    /// calls `last` right away (the caller snapshots it mid-run and after).
    private func replaySequence(_ files: [String], index: Int, app: AppDelegate, last: @escaping () -> Void) {
        let f = files[index]
        note("replay \(index + 1)/\(files.count) \(f)")
        app.web.evaluateJavaScript("window.__hs.replay(\(Self.js(f))); true") { _, e in
            if let e = e { self.note("replay error \(e)") }
        }
        if index == files.count - 1 { return last() }
        var waited = 0.0
        func poll() {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                waited += 0.5
                app.web.evaluateJavaScript("!!(window.__hs.run && window.__hs.run.running)") { r, _ in
                    let running = (r as? Bool) ?? false
                    if (!running && waited > 2) || waited > 40 {
                        self.note("replay \(index + 1) ended after \(waited) s\(waited > 40 ? " (TIMEOUT)" : "")")
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                            self.replaySequence(files, index: index + 1, app: app, last: last)
                        }
                    } else { poll() }
                }
            }
        }
        poll()
    }

    private func micSequence(_ holds: [Double], index: Int, app: AppDelegate) {
        guard index < holds.count else {
            // let the last run's stale events arrive, then confirm the page is idle
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                app.web.evaluateJavaScript("JSON.stringify(window.__hs.mic.stats())") { r, _ in
                    self.note("mic final \((r as? String) ?? "?")")
                    self.finish(app)
                }
            }
            return
        }
        let h = holds[index]
        note("mic press \(index + 1)/\(holds.count) hold \(h) s")
        app.web.evaluateJavaScript("window.__hs.mic.start(false); true") { _, e in if let e = e { self.note("mic start error \(e)") } }
        // sample mid-hold too: a capture that stalls shows frames flat while held
        DispatchQueue.main.asyncAfter(deadline: .now() + h / 2) {
            app.web.evaluateJavaScript("JSON.stringify(window.__hs.mic.stats())") { r, _ in
                self.note("mic mid \(index + 1) \((r as? String) ?? "?")")
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + h) {
            app.web.evaluateJavaScript("(() => { const on = window.__hs.mic.on; const s = window.__hs.mic.stats(); window.__hs.mic.stop(); return JSON.stringify({still_on: on, ...s}) })()") { r, _ in
                self.note("mic release \(index + 1) hold \(h) s: \((r as? String) ?? "?")")
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { self.micSequence(holds, index: index + 1, app: app) }
            }
        }
    }

    private func pageState(_ app: AppDelegate, _ tag: String, then: @escaping () -> Void) {
        let js = "JSON.stringify({focus: document.activeElement && (document.activeElement.id || document.activeElement.tagName), " +
            "dom: document.getElementsByTagName('*').length, running: !!(window.__hs.run && window.__hs.run.running), " +
            "kind: window.__hs.run && window.__hs.run.kind, film: !!(window.HSFilm && window.HSFilm.active), " +
            "answers: window.__hs.run ? Object.keys(window.__hs.run.q.answer || {}).filter(k => (window.__hs.run.q.answer[k] || '').length > 0).length : 0, " +
            "box: (document.getElementById('typeBox') || {}).value})"
        app.web.evaluateJavaScript(js) { r, e in
            self.note("\(tag) \((r as? String) ?? "error \(String(describing: e))")")
            then()
        }
    }

    private func typeChars(_ chars: [Character], _ i: Int, app: AppDelegate, done: @escaping () -> Void) {
        guard i < chars.count else { return done() }
        if autoTypeKeys { key(app, String(chars[i]), code: chars[i] == " " ? 49 : 0) } else { app.web.insertText(String(chars[i])) }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05 + Double.random(in: 0...0.09)) {
            self.typeChars(chars, i + 1, app: app, done: done)
        }
    }

    private func typeSequence(_ qs: [String], index: Int, app: AppDelegate) {
        guard index < qs.count else {
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                self.pageState(app, "type final") { self.saveWeb(app.web, "06_typed_done.png") { self.finish(app) } }
            }
            return
        }
        let q = qs[index]
        note("type \(index + 1)/\(qs.count) \(q)")
        // focus the box the way a person does (the "/" shortcut), select what is there, then type
        app.web.evaluateJavaScript("(() => { const b = document.getElementById('typeBox'); b.focus(); b.select(); return document.activeElement.id })()") { _, _ in
            self.typeChars(Array(q), 0, app: app) {
                let spam = index % 2 == 1 ? 6 : 1
                if self.autoTypeKeys { for _ in 0..<spam { self.key(app, "\r", code: 36) } }
                let enter = self.autoTypeKeys ? "true"
                    : "(() => { const f = document.getElementById('typeForm'); for (let k = 0; k < \(spam); k++) f.requestSubmit(); return true })()"
                app.web.evaluateJavaScript(enter) { _, _ in
                    self.pageState(app, "type \(index + 1) enter") {
                        let next = { (delay: Double) in
                            var waited = 0.0
                            func poll() {
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                    waited += 0.5
                                    app.web.evaluateJavaScript("!!(window.__hs.run && window.__hs.run.running)") { r, _ in
                                        if ((r as? Bool) ?? false) && waited < 30 { return poll() }
                                        self.pageState(app, "type \(index + 1) answered after \(waited) s\(waited >= 30 ? " (TIMEOUT)" : "")") {
                                            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { self.typeSequence(qs, index: index + 1, app: app) }
                                        }
                                    }
                                }
                            }
                            poll()
                        }
                        // every third question: the next one starts while this one is still answering
                        if index % 3 == 2 { DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { self.typeSequence(qs, index: index + 1, app: app) } }
                        else { next(0.5) }
                    }
                }
            }
        }
    }

    private func finish(_ app: AppDelegate) {
        note("done")
        if autoQuit { NSApp.perform(#selector(NSApplication.terminate(_:)), with: nil, afterDelay: 1.0) }
    }

    private static func js(_ s: String) -> String {
        let d = try? JSONSerialization.data(withJSONObject: [s])
        return String((String(data: d ?? Data(), encoding: .utf8) ?? "[\"\"]").dropFirst().dropLast())
    }

    private func saveView(_ v: NSView, _ name: String) {
        guard let d = dir, let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) else { return }
        v.cacheDisplay(in: v.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: d.appendingPathComponent(name))
        note("saved \(name)")
    }

    private func saveWeb(_ w: WKWebView, _ name: String, then: @escaping () -> Void) {
        guard let d = dir else { return then() }
        w.takeSnapshot(with: nil) { img, err in
            if let img = img, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) {
                try? rep.representation(using: .png, properties: [:])?.write(to: d.appendingPathComponent(name))
                self.note("saved \(name)")
            } else {
                self.note("snapshot \(name) failed: \(String(describing: err))")
            }
            then()
        }
    }
}
