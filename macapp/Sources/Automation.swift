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
    private var splashShot = false
    private var pageDone = false
    private var log: [String] = []

    private func note(_ s: String) {
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
        guard dir != nil, !pageDone else { return }
        pageDone = true
        note("page loaded")
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            self.saveWeb(app.web, "02_ui_loaded.png") {
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
