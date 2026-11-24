import AppKit
import SwiftUI
import WebKit

/// The native shell: one window around the same web UI, served by the bundled local server.
final class AppDelegate: NSObject, NSApplicationDelegate, WKUIDelegate, WKNavigationDelegate, NSWindowDelegate,
                         WKScriptMessageHandler {
    let server = ServerManager()
    let splash = SplashModel()
    var window: NSWindow!
    var web: WKWebView!
    var splashHost: NSHostingView<SplashView>!
    var settingsWindow: NSWindow?
    var loadedPort = 0
    var loadedGeneration = 0
    var quitting = false
    var micActivity: NSObjectProtocol?
    static let hidden = UserDefaults.standard.bool(forKey: "hidden")
    /// The Developer menu (race view, replay questions, keys, server restart) stays out of the consumer app: shown
    /// when Option is held at launch or after `defaults write <bundle id> developerMenu -bool true`.
    static let developerMenu = UserDefaults.standard.bool(forKey: "developerMenu") || NSEvent.modifierFlags.contains(.option)
    /// Developer view: the engine page (web/lab.html: race, timeline, query frame) instead of the consumer page.
    var devView = UserDefaults.standard.bool(forKey: "developerView") && AppDelegate.developerMenu
    func pageURL(_ port: Int) -> URL { URL(string: "http://127.0.0.1:\(port)/" + (devView ? "lab.html" : ""))! }

    func applicationDidFinishLaunching(_ note: Notification) {
        buildMenu()
        // a rebuilt app may reuse last launch's port: drop cached pages/scripts (localStorage, the memory id, stays)
        let cacheTypes: Set<String> = [WKWebsiteDataTypeDiskCache, WKWebsiteDataTypeMemoryCache]
        WKWebsiteDataStore.default().removeData(ofTypes: cacheTypes, modifiedSince: .distantPast) {}
        buildWindow()
        server.onState = { [weak self] s in self?.apply(s) }
        if !Self.hidden { NSApp.activate(ignoringOtherApps: true) }
        // no key screen on first run: Onset works with no keys (on-device answers, DuckDuckGo). Your own service
        // keys are optional, under Settings > Advanced.
        AppInfo.onboardingDone = true
        server.start()
        Updater.shared.checkAtLaunch()       // GitHub releases, at most once a day (Updater.swift)
    }

    // MARK: onboarding (keys)

    var onboardingSheet: NSWindow?

    @objc func setUpKeys() { showOnboarding(firstRun: false) }

    func showOnboarding(firstRun: Bool) {
        guard onboardingSheet == nil else { return }
        let sheet = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 620, height: 720), styleMask: [.titled],
                             backing: .buffered, defer: false)
        let host = NSHostingView(rootView: OnboardingView(onDone: { [weak self] changed in
            guard let self = self, let sh = self.onboardingSheet else { return }
            self.window.endSheet(sh)
            self.onboardingSheet = nil
            if firstRun { self.server.start() } else if changed { self.loadedPort = 0; self.server.restart() }
        }))
        sheet.contentView = host
        onboardingSheet = sheet
        window.beginSheet(sheet)
        Automation.shared.onboardingShown(host, app: self)
    }

    /// Automation (-autoOnboarding skip): press "Skip: use the free mode".
    func skipOnboarding() {
        guard let sh = onboardingSheet else { return }
        AppInfo.onboardingDone = true
        window.endSheet(sh)
        onboardingSheet = nil
        server.start()
    }

    // MARK: window

    func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 960, height: 780),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.title = AppInfo.name
        window.minSize = NSSize(width: 820, height: 560)
        if !Self.hidden {
            window.setFrameAutosaveName("MainWindow")
            if !window.setFrameUsingName("MainWindow") { window.center() }
        }
        window.delegate = self
        window.isReleasedWhenClosed = false

        let cfg = WKWebViewConfiguration()
        cfg.mediaTypesRequiringUserActionForPlayback = []
        cfg.preferences.setValue(true, forKey: "developerExtrasEnabled")
        // page -> app bridge (window.webkit.messageHandlers.onset): {t: "settings"} from the notice line,
        // {t: "mic", on} while the page listens (App Nap off),
        // {t: "savePNG", name, data: "data:image/png;base64,..."} from the Results view's "Save as image"
        cfg.userContentController.add(self, name: "onset")
        web = WKWebView(frame: .zero, configuration: cfg)
        web.uiDelegate = self
        web.navigationDelegate = self
        if #available(macOS 13.3, *) { web.isInspectable = true }
        web.setValue(false, forKey: "drawsBackground")
        web.isHidden = true

        splashHost = NSHostingView(rootView: SplashView(
            model: splash,
            onDownload: { [weak self] in self?.server.downloadModels() },
            onImport: { [weak self] in self?.importModels() },
            onOpenLogs: { [weak self] in self?.openLogs() },
            onRetry: { [weak self] in self?.server.restart() },
            onSettings: { [weak self] in self?.showSettings() }))

        let root = NSView()
        for v in [web!, splashHost!] as [NSView] {
            v.translatesAutoresizingMaskIntoConstraints = false
            root.addSubview(v)
            NSLayoutConstraint.activate([
                v.leadingAnchor.constraint(equalTo: root.leadingAnchor), v.trailingAnchor.constraint(equalTo: root.trailingAnchor),
                v.topAnchor.constraint(equalTo: root.topAnchor), v.bottomAnchor.constraint(equalTo: root.bottomAnchor)])
        }
        window.contentView = root
        if Self.hidden {
            // test instance (-hidden YES): no Dock icon, never activated, window parked off-screen (it still renders,
            // so the automation snapshots work) — it does not take over the user's screen
            window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
            window.orderBack(nil)
        } else {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func apply(_ s: ServerManager.State) {
        splash.state = s
        switch s {
        case .ready(let port):
            // a new server (first start, restart after a crash, Settings saved): always reload the page, so the
            // websocket, the question list and the warm state come from this server
            if loadedPort != port || web.url == nil || loadedGeneration != server.generation {
                loadedPort = port
                loadedGeneration = server.generation
                web.load(URLRequest(url: pageURL(port)))
            } else {
                showWeb(true)
            }
        default:
            showWeb(false)
        }
        Automation.shared.stateChanged(s, app: self)
    }

    func showWeb(_ on: Bool) {
        web.isHidden = !on
        splashHost.isHidden = on
        if on { window.makeFirstResponder(web) }   // Space (push-to-talk) goes to the page
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        if case .ready = server.state { showWeb(true) }
        showPendingNotice()
        loadReplayMenu()
        Automation.shared.pageLoaded(app: self)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { retryLoad() }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { retryLoad() }
    /// The page's web content process ended (it crashed, or macOS reclaimed its memory). The app and its server keep
    /// running: log it, reload the page against the same server and say so in a plain notice on the reloaded page,
    /// instead of leaving a blank window. More than 3 in a minute: stop reloading and ask (no reload loop).
    private var pageEnds: [Date] = []
    private var pendingNotice: String?
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        let now = Date()
        pageEnds = pageEnds.filter { now.timeIntervalSince($0) < 60 } + [now]
        appLog("page content process ended (\(pageEnds.count) in the last minute); "
               + (pageEnds.count > 3 ? "not reloading again" : "reloading the page"))
        Automation.shared.note("page content process ended")
        guard pageEnds.count <= 3 else {
            showWeb(false)
            let a = NSAlert()
            a.messageText = "The page keeps stopping"
            a.informativeText = "\(AppInfo.name)'s window stopped \(pageEnds.count) times in a minute. The local server is still running. Reload to try again, or open the logs."
            a.addButton(withTitle: "Reload")
            a.addButton(withTitle: "Open Logs")
            a.beginSheetModal(for: window) { [weak self] r in
                guard let self = self else { return }
                if r == .alertSecondButtonReturn { self.openLogs() }
                self.pageEnds.removeAll()
                self.reloadAfterEnd()
            }
            return
        }
        reloadAfterEnd()
    }

    private func reloadAfterEnd() {
        pendingNotice = "The page stopped unexpectedly and was reloaded. Your conversation memory is kept; ask again."
        if case .ready(let port) = server.state {
            web.load(URLRequest(url: pageURL(port)))
        } else {
            web.reload()
        }
    }

    /// After a reload that followed a page crash: the plain notice on the page (web/app.js window.__hsNotice).
    private func showPendingNotice() {
        guard let n = pendingNotice else { return }
        pendingNotice = nil
        let d = (try? JSONSerialization.data(withJSONObject: [n])).flatMap { String(data: $0, encoding: .utf8) } ?? "[\"\"]"
        web.evaluateJavaScript("window.__hsNotice && window.__hsNotice(\(d)[0]); true") { _, _ in }
    }

    private func retryLoad() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            guard let self = self, case .ready(let port) = self.server.state else { return }
            self.web.load(URLRequest(url: self.pageURL(port)))
        }
    }

    // MARK: WKUIDelegate: microphone + links + JS dialogs

    private func isOurs(_ origin: WKSecurityOrigin) -> Bool {
        (origin.host == "127.0.0.1" || origin.host == "localhost") && origin.port == loadedPort
    }

    /// getUserMedia: grant the page's own origin (macOS still asks the user once, via NSMicrophoneUsageDescription).
    @available(macOS 12.0, *)
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(isOurs(origin) && type == .microphone ? .grant : .deny)
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let u = action.request.url, let host = u.host, host != "127.0.0.1", host != "localhost",
           ["http", "https"].contains(u.scheme ?? "") {
            NSWorkspace.shared.open(u)            // source links open in the default browser
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let u = action.request.url { NSWorkspace.shared.open(u) }
        return nil
    }

    /// <input type="file"> (WKWebView shows no picker unless the app does it): audio / video files for My recordings
    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { r in completionHandler(r == .OK ? panel.urls : nil) }
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let a = NSAlert(); a.messageText = AppInfo.name; a.informativeText = message
        a.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let a = NSAlert(); a.messageText = AppInfo.name; a.informativeText = message
        a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
        a.beginSheetModal(for: window) { r in completionHandler(r == .alertFirstButtonReturn) }
    }

    // MARK: page bridge

    func userContentController(_ ucc: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "onset", isOurs(message.frameInfo.securityOrigin),
              let body = message.body as? [String: Any], let t = body["t"] as? String else { return }
        switch t {
        case "settings":
            showSettings()
        case "log":
            // the page's own uncaught errors (web/app.js): one line each in logs/app.log, so a "crash" can be traced
            if let m = body["msg"] as? String { appLog("page: " + String(m.prefix(400))) }
        case "mic":
            // listening: keep the app out of App Nap / timer throttling (it may sit behind other windows while held)
            let on = (body["on"] as? Bool) ?? false
            if on, micActivity == nil {
                micActivity = ProcessInfo.processInfo.beginActivity(
                    options: [.userInitiated, .latencyCritical], reason: "Listening to the microphone")
            } else if !on, let a = micActivity {
                ProcessInfo.processInfo.endActivity(a)
                micActivity = nil
            }
        case "open":
            // a file from a "find my …" result: opened with its default app. Anything that would run code (apps,
            // scripts, installers) is only shown in Finder.
            guard let path = body["path"] as? String, path.hasPrefix(NSHomeDirectory() + "/") || path.hasPrefix("/Volumes/"),
                  FileManager.default.fileExists(atPath: path) else { return }
            let url = URL(fileURLWithPath: path).standardizedFileURL
            let risky: Set<String> = ["app", "command", "sh", "zsh", "bash", "tool", "pkg", "mpkg", "terminal", "workflow",
                                      "scpt", "applescript", "jar", "py", "rb", "pl", "dmg", "action", "prefpane", "webloc", "inetloc"]
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
            if risky.contains(url.pathExtension.lowercased()) || (!isDir.boolValue && FileManager.default.isExecutableFile(atPath: path)) {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } else {
                NSWorkspace.shared.open(url)
            }
        case "pickRecordings":
            // My recordings (web/recordings.js): pick audio / video files; the server reads them in place by path
            // (no upload copy of a long recording), via HSRecordings.addPaths
            let panel = NSOpenPanel()
            panel.allowsMultipleSelection = true
            panel.canChooseDirectories = false
            panel.allowedContentTypes = [.audio, .movie]
            panel.message = "Choose recordings to index on this Mac"
            panel.beginSheetModal(for: window) { [weak self] r in
                guard r == .OK, let self = self else { return }
                let paths = panel.urls.map { $0.path }
                guard let data = try? JSONSerialization.data(withJSONObject: paths),
                      let json = String(data: data, encoding: .utf8) else { return }
                self.web.evaluateJavaScript("window.HSRecordings && window.HSRecordings.addPaths(\(json))")
            }
        case "savePNG":
            guard let s = body["data"] as? String, let comma = s.firstIndex(of: ","), s.hasPrefix("data:image/png;base64"),
                  let png = Data(base64Encoded: String(s[s.index(after: comma)...])) else { return }
            let panel = NSSavePanel()
            panel.nameFieldStringValue = (body["name"] as? String).map { ($0 as NSString).lastPathComponent } ?? "Onset results.png"
            panel.allowedContentTypes = [.png]
            panel.beginSheetModal(for: window) { r in
                guard r == .OK, let url = panel.url else { return }
                try? png.write(to: url)
            }
        default:
            break
        }
    }

    // MARK: menu actions

    @objc func showSettings() {
        if settingsWindow == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 540, height: 360),
                             styleMask: [.titled, .closable], backing: .buffered, defer: false)
            w.title = "\(AppInfo.name) Settings"
            w.isReleasedWhenClosed = false
            w.contentView = NSHostingView(rootView: SettingsView(onSave: { [weak self] in
                self?.loadedPort = 0
                self?.server.restart()
            }))
            w.center()
            settingsWindow = w
        }
        settingsWindow?.makeKeyAndOrderFront(nil)
    }

    @objc func forgetMemory() {
        let a = NSAlert()
        a.messageText = "Forget memory?"
        a.informativeText = "Clears what \(AppInfo.name) remembers from your conversations (people, places, your facts), in this window and on disk."
        a.addButton(withTitle: "Forget"); a.addButton(withTitle: "Cancel")
        a.beginSheetModal(for: window) { [weak self] r in
            guard r == .alertFirstButtonReturn, let self = self else { return }
            self.web.evaluateJavaScript("(() => { const b = document.getElementById('memForget'); if (b) { b.click(); return true } return false })()") { res, _ in
                if (res as? Bool) != true {
                    // page not connected: clear the on-disk store directly
                    try? FileManager.default.removeItem(at: AppInfo.dataDir.appendingPathComponent("memory"))
                }
            }
        }
    }

    @objc func clearCache() {
        let a = NSAlert()
        a.messageText = "Clear search cache?"
        a.informativeText = "Forgets earlier search results that \(AppInfo.name) reuses across launches. The next questions search live again."
        a.addButton(withTitle: "Clear"); a.addButton(withTitle: "Cancel")
        a.beginSheetModal(for: window) { [weak self] r in
            guard r == .alertFirstButtonReturn, let self = self else { return }
            if case .ready(let port) = self.server.state, let u = URL(string: "http://127.0.0.1:\(port)/api/cache/clear") {
                var req = URLRequest(url: u); req.httpMethod = "POST"
                URLSession.shared.dataTask(with: req).resume()
            } else {   // server not running: remove the file directly
                try? FileManager.default.removeItem(at: AppInfo.dataDir.appendingPathComponent("search_cache.json"))
            }
        }
    }

    @objc func openLogs() {
        try? FileManager.default.createDirectory(at: AppInfo.logsDir, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: AppInfo.serverLog.path) {
            NSWorkspace.shared.open(AppInfo.serverLog)
        } else {
            NSWorkspace.shared.open(AppInfo.logsDir)
        }
    }

    // MARK: replay menu + details (the page's default "minimal" view hides its question list; D or ⌘D shows it)

    let replayMenu = NSMenu(title: "Replay")

    func loadReplayMenu() {
        guard loadedPort > 0, let u = URL(string: "http://127.0.0.1:\(loadedPort)/api/questions") else { return }
        URLSession.shared.dataTask(with: u) { [weak self] data, _, _ in
            guard let self = self, let d = data,
                  let rows = try? JSONSerialization.jsonObject(with: d) as? [[String: Any]] else { return }
            DispatchQueue.main.async { self.fillReplayMenu(rows) }
        }.resume()
    }

    func fillReplayMenu(_ rows: [[String: Any]]) {
        replayMenu.removeAllItems()
        let groups: [(String, ([String: Any]) -> Bool)] = [
            ("Web questions", { ($0["kind"] as? String) != "local" && $0["parts"] == nil }),
            ("Multi-part", { $0["parts"] != nil }),
            ("Local moments", { ($0["kind"] as? String) == "local" })]
        var n = 0
        for (title, keep) in groups {
            let items = rows.filter(keep)
            if items.isEmpty { continue }
            let head = NSMenuItem(title: title, action: nil, keyEquivalent: ""); head.isEnabled = false
            replayMenu.addItem(head)
            let sub = NSMenu(title: title)
            for r in items {
                guard let f = r["file"] as? String else { continue }
                let text = (r["text"] as? String) ?? f
                let it = NSMenuItem(title: text, action: #selector(replayItem(_:)), keyEquivalent: "")
                it.representedObject = f
                it.target = self
                if title == "Web questions" && n < 9 { n += 1; it.keyEquivalent = String(n); it.keyEquivalentModifierMask = [.command, .option] }
                (items.count > 12 ? sub : replayMenu).addItem(it)
            }
            if items.count > 12 {
                let more = NSMenuItem(title: "All \(items.count)", action: nil, keyEquivalent: ""); more.submenu = sub
                replayMenu.addItem(more)
            }
            replayMenu.addItem(.separator())
        }
        if replayMenu.items.isEmpty {
            let none = NSMenuItem(title: "No replay questions (demo-audio not installed)", action: nil, keyEquivalent: "")
            none.isEnabled = false
            replayMenu.addItem(none)
        }
    }

    @objc func replayItem(_ sender: NSMenuItem) {
        guard let f = sender.representedObject as? String,
              let d = try? JSONSerialization.data(withJSONObject: [f]), let arr = String(data: d, encoding: .utf8) else { return }
        web.evaluateJavaScript("window.__hs && window.__hs.replay(\(arr)[0]); true", completionHandler: nil)
    }

    @objc func toggleDetails() {
        // Developer view: the engine page (race, timeline, question list) vs the consumer page
        devView.toggle()
        UserDefaults.standard.set(devView, forKey: "developerView")
        if case .ready(let port) = server.state { web.load(URLRequest(url: pageURL(port))) }
    }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        if item.action == #selector(toggleDetails) { item.state = devView ? .on : .off }
        return true
    }

    @objc func openModels() { NSWorkspace.shared.open(AppInfo.modelsDir) }
    @objc func restartServer() { loadedPort = 0; server.restart() }
    @objc func reloadPage() { appLog("page reloaded (View > Reload)"); web.reload() }

    @objc func showAbout() {
        let credits = NSAttributedString(string: """
            Ask out loud. The answer is ready when you stop talking.
            Listening and answers run on this Mac. Web results come from DuckDuckGo.
            Built with Arc Mini and Arc Tagger (Whissle, CC BY-NC 4.0), EmbeddingGemma (Google, Apache 2.0) and \
            Qwen2.5-0.5B-Instruct (Qwen, Apache 2.0). Licences: \(AppInfo.name).app/Contents/Resources/Licenses
            """, attributes: [.font: NSFont.systemFont(ofSize: 11), .foregroundColor: NSColor.secondaryLabelColor])
        NSApp.orderFrontStandardAboutPanel(options: [.credits: credits, .applicationName: AppInfo.name])
    }

    func importModels() {
        guard case .needsModels(let missing) = server.state else { return }
        let p = NSOpenPanel()
        p.canChooseDirectories = true; p.canChooseFiles = false; p.allowsMultipleSelection = false
        p.message = "Choose a folder holding the models to import (" + missing.filter { !$0.downloadable }.map { $0.model }.joined(separator: ", ") + ")"
        p.beginSheetModal(for: window) { [weak self] r in
            guard r == .OK, let u = p.url, let self = self else { return }
            DispatchQueue.global().async { _ = self.server.importModels(from: u, rows: missing) }
        }
    }

    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let app = NSMenu()
        app.addItem(withTitle: "About \(AppInfo.name)", action: #selector(showAbout), keyEquivalent: "")
        let upd = NSMenuItem(title: "Check for Updates…", action: #selector(Updater.checkNow), keyEquivalent: "")
        upd.target = Updater.shared; app.addItem(upd)
        app.addItem(.separator())
        app.addItem(withTitle: "Settings…", action: #selector(showSettings), keyEquivalent: ",")
        app.addItem(withTitle: "Forget What Onset Remembers…", action: #selector(forgetMemory), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "Hide \(AppInfo.name)", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        app.addItem(.separator())
        app.addItem(withTitle: "Quit \(AppInfo.name)", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = app

        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit

        let viewItem = NSMenuItem(); main.addItem(viewItem)
        let view = NSMenu(title: "View")
        view.addItem(withTitle: "Reload", action: #selector(reloadPage), keyEquivalent: "r")
        view.addItem(withTitle: "Enter Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
            .keyEquivalentModifierMask = [.command, .control]
        viewItem.submenu = view

        if AppDelegate.developerMenu {
            let devItem = NSMenuItem(); main.addItem(devItem)
            let dev = NSMenu(title: "Developer")
            dev.addItem(withTitle: "Developer View", action: #selector(toggleDetails), keyEquivalent: "d")
                .keyEquivalentModifierMask = [.command, .option]
            let replayTop = NSMenuItem(title: "Replay", action: nil, keyEquivalent: "")
            let loading = NSMenuItem(title: "Loading…", action: nil, keyEquivalent: ""); loading.isEnabled = false
            replayMenu.addItem(loading)
            replayTop.submenu = replayMenu
            dev.addItem(replayTop)
            dev.addItem(.separator())
            dev.addItem(withTitle: "Set Up Keys…", action: #selector(setUpKeys), keyEquivalent: "")
            dev.addItem(withTitle: "Clear Search Cache…", action: #selector(clearCache), keyEquivalent: "")
            dev.addItem(withTitle: "Open Models Folder", action: #selector(openModels), keyEquivalent: "")
            dev.addItem(withTitle: "Restart Server", action: #selector(restartServer), keyEquivalent: "")
            devItem.submenu = dev
        }

        let winItem = NSMenuItem(); main.addItem(winItem)
        let win = NSMenu(title: "Window")
        win.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        winItem.submenu = win
        NSApp.windowsMenu = win
        let helpItem = NSMenuItem(); main.addItem(helpItem)
        let help = NSMenu(title: "Help")
        help.addItem(withTitle: "Open Logs (for a bug report)", action: #selector(openLogs), keyEquivalent: "")
        helpItem.submenu = help
        NSApp.helpMenu = help
        NSApp.mainMenu = main
    }

    // MARK: lifecycle

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if quitting { return .terminateNow }
        quitting = true
        DispatchQueue.global().async {
            self.server.stop()
            // reply through the run loop, not the main dispatch queue: terminate: may have been called from a main-queue
            // block, and .terminateLater then waits in a nested run loop while that block still holds the queue
            self.performSelector(onMainThread: #selector(self.replyTerminate), with: nil, waitUntilDone: false,
                                 modes: [RunLoop.Mode.common.rawValue, RunLoop.Mode.modalPanel.rawValue,
                                         RunLoop.Mode.eventTracking.rawValue])
        }
        return .terminateLater
    }

    @objc func replyTerminate() { NSApp.reply(toApplicationShouldTerminate: true) }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(AppDelegate.hidden ? .accessory : .regular)
app.run()
