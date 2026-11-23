import AppKit
import Foundation
import Security

/// Updates from GitHub releases (WhissleAI/onset). Once a day at launch (Settings default on) and from
/// "Check for Updates…": reads releases/latest, and when its tag (vX.Y.Z) is newer than this build offers to install it.
/// Install: download the release DMG and its .sha256, check the hash, mount it read-only, check that the app inside is
/// validly signed by the SAME Developer ID team as this app and has this bundle id, then quit; a small helper replaces
/// this app with the new one (same path) and relaunches it. Anything unexpected (no DMG asset, hash or signature
/// mismatch, unwritable install folder) stops before touching this app and offers the release page instead.
/// Developer (ad-hoc signed) builds never install automatically: they only say a newer release exists.
final class Updater: NSObject {
    static let shared = Updater()
    static let repo = "WhissleAI/onset"
    static var releasesPage: URL { URL(string: "https://github.com/\(repo)/releases/latest")! }
    private static let api = URL(string: "https://api.github.com/repos/\(repo)/releases/latest")!
    private static let lastCheckKey = "updateLastCheck", skipKey = "updateSkipVersion", autoKey = "updateAutoCheck"

    static var autoCheck: Bool {
        get { UserDefaults.standard.object(forKey: autoKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: autoKey) }
    }

    private var busy = false

    struct Release { let version: String; let page: URL; let dmg: URL?; let sha: URL?; let notes: String }

    // MARK: checking

    /// At launch: at most once a day, quiet when there is nothing new or the network is down.
    func checkAtLaunch() {
        guard Self.autoCheck, AppInfo.supportOverride == nil else { return }
        let last = UserDefaults.standard.double(forKey: Self.lastCheckKey)
        guard Date().timeIntervalSince1970 - last > 20 * 3600 else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 20) { self.check(userInitiated: false) }
    }

    @objc func checkNow() { check(userInitiated: true) }

    func check(userInitiated: Bool) {
        guard !busy else { return }
        busy = true
        var req = URLRequest(url: Self.api, timeoutInterval: 15)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("\(AppInfo.name)/\(AppInfo.version) (macOS updater)", forHTTPHeaderField: "User-Agent")
        URLSession.shared.dataTask(with: req) { data, resp, err in
            let rel = data.flatMap { Self.parse($0) }
            DispatchQueue.main.async {
                self.busy = false
                UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: Self.lastCheckKey)
                guard let rel = rel else {
                    appLog("update check failed: \(err?.localizedDescription ?? "HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0)")")
                    if userInitiated { self.info("Couldn't check for updates", "GitHub didn't answer. Try again later, or open the releases page.", page: true) }
                    return
                }
                appLog("update check: latest \(rel.version), this \(AppInfo.version)")
                if Self.isNewer(rel.version, than: AppInfo.version) {
                    if !userInitiated && UserDefaults.standard.string(forKey: Self.skipKey) == rel.version { return }
                    self.offer(rel)
                } else if userInitiated {
                    self.info("\(AppInfo.name) is up to date", "You have version \(AppInfo.version), the latest release.", page: false)
                }
            }
        }.resume()
    }

    static func parse(_ data: Data) -> Release? {
        guard let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tag = j["tag_name"] as? String, (j["draft"] as? Bool) != true, (j["prerelease"] as? Bool) != true,
              let page = (j["html_url"] as? String).flatMap(URL.init(string:)) else { return nil }
        let assets = (j["assets"] as? [[String: Any]]) ?? []
        func asset(_ suffix: String) -> URL? {
            assets.first { ($0["name"] as? String)?.hasSuffix(suffix) == true }
                .flatMap { $0["browser_download_url"] as? String }.flatMap(URL.init(string:))
        }
        let v = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
        return Release(version: v, page: page, dmg: asset("-arm64.dmg") ?? asset(".dmg"), sha: asset(".dmg.sha256"),
                       notes: (j["body"] as? String) ?? "")
    }

    /// Numeric dotted-version compare: "0.10.0" > "0.9.3"; a missing part counts as 0; text after a "-" is ignored.
    static func isNewer(_ a: String, than b: String) -> Bool {
        func parts(_ s: String) -> [Int] {
            s.split(separator: "-").first.map { $0.split(separator: ".").map { Int($0) ?? 0 } } ?? []
        }
        let x = parts(a), y = parts(b)
        for i in 0..<max(x.count, y.count) {
            let p = i < x.count ? x[i] : 0, q = i < y.count ? y[i] : 0
            if p != q { return p > q }
        }
        return false
    }

    // MARK: offering

    private func offer(_ rel: Release) {
        let a = NSAlert()
        a.messageText = "\(AppInfo.name) \(rel.version) is available"
        let notes = rel.notes.split(separator: "\n").prefix(8).joined(separator: "\n")
        a.informativeText = "You have \(AppInfo.version).\(notes.isEmpty ? "" : "\n\n" + notes)"
        let canInstall = rel.dmg != nil && rel.sha != nil && Self.ownTeamID() != nil
        a.addButton(withTitle: canInstall ? "Install and Relaunch" : "Open Download Page")
        a.addButton(withTitle: "Later")
        a.addButton(withTitle: "Skip This Version")
        switch a.runModal() {
        case .alertFirstButtonReturn:
            if canInstall { install(rel) } else { NSWorkspace.shared.open(rel.page) }
        case .alertThirdButtonReturn:
            UserDefaults.standard.set(rel.version, forKey: Self.skipKey)
        default: break
        }
    }

    private func info(_ title: String, _ text: String, page: Bool) {
        let a = NSAlert(); a.messageText = title; a.informativeText = text
        a.addButton(withTitle: "OK"); if page { a.addButton(withTitle: "Open Releases Page") }
        if a.runModal() == .alertSecondButtonReturn { NSWorkspace.shared.open(Self.releasesPage) }
    }

    private func fail(_ why: String, _ rel: Release) {
        appLog("update \(rel.version) not installed: \(why)")
        let a = NSAlert(); a.messageText = "The update wasn't installed"
        a.informativeText = "\(why)\n\nYour current version is unchanged. You can download the update from the releases page."
        a.addButton(withTitle: "Open Releases Page"); a.addButton(withTitle: "Close")
        if a.runModal() == .alertFirstButtonReturn { NSWorkspace.shared.open(rel.page) }
    }

    // MARK: installing

    private var progress: NSWindow?

    private func showProgress(_ text: String) {
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 340, height: 90), styleMask: [.titled], backing: .buffered, defer: false)
        w.title = "Updating \(AppInfo.name)"
        let label = NSTextField(labelWithString: text); label.frame = NSRect(x: 20, y: 50, width: 300, height: 20)
        let bar = NSProgressIndicator(frame: NSRect(x: 20, y: 20, width: 300, height: 20))
        bar.isIndeterminate = true; bar.startAnimation(nil)
        w.contentView?.addSubview(label); w.contentView?.addSubview(bar)
        w.center(); w.makeKeyAndOrderFront(nil); progress = w
    }

    private func install(_ rel: Release) {
        guard let dmgURL = rel.dmg, let shaURL = rel.sha else { return }
        let target = Bundle.main.bundleURL
        let parent = target.deletingLastPathComponent()
        guard FileManager.default.isWritableFile(atPath: parent.path) else {
            fail("\(parent.path) isn't writable by your account.", rel); return
        }
        showProgress("Downloading \(AppInfo.name) \(rel.version)…")
        let work = FileManager.default.temporaryDirectory.appendingPathComponent("onset-update-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        DispatchQueue.global(qos: .userInitiated).async {
            let result = Result { try self.prepare(dmgURL: dmgURL, shaURL: shaURL, work: work) }
            DispatchQueue.main.async {
                self.progress?.close(); self.progress = nil
                switch result {
                case .failure(let e): try? FileManager.default.removeItem(at: work); self.fail("\(e)", rel)
                case .success(let newApp): self.swapAndRelaunch(newApp: newApp, target: target, work: work, version: rel.version)
                }
            }
        }
    }

    struct UpdateError: Error, CustomStringConvertible { let description: String }

    /// Download + verify; returns the verified new .app copied out of the DMG into `work`.
    private func prepare(dmgURL: URL, shaURL: URL, work: URL) throws -> URL {
        let dmg = work.appendingPathComponent("update.dmg")
        try download(dmgURL, to: dmg)
        let shaFile = work.appendingPathComponent("update.sha256")
        try download(shaURL, to: shaFile)
        let want = (try String(contentsOf: shaFile, encoding: .utf8)).split(whereSeparator: { $0 == " " || $0 == "\n" }).first.map(String.init)?.lowercased() ?? ""
        let got = try run("/usr/bin/shasum", ["-a", "256", dmg.path]).split(separator: " ").first.map(String.init) ?? ""
        guard want.count == 64, want == got else { throw UpdateError(description: "The download's checksum doesn't match the release (\(got.prefix(12))… vs \(want.prefix(12))…).") }
        let mnt = work.appendingPathComponent("mnt")
        _ = try run("/usr/bin/hdiutil", ["attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", mnt.path, dmg.path])
        defer { _ = try? run("/usr/bin/hdiutil", ["detach", "-force", mnt.path]) }
        let found = (try FileManager.default.contentsOfDirectory(at: mnt, includingPropertiesForKeys: nil))
            .first { $0.pathExtension == "app" }
        guard let src = found else { throw UpdateError(description: "The disk image has no app in it.") }
        let newApp = work.appendingPathComponent(src.lastPathComponent)
        _ = try run("/usr/bin/ditto", [src.path, newApp.path])
        try Self.verify(newApp)
        return newApp
    }

    private func download(_ url: URL, to dest: URL) throws {
        let sem = DispatchSemaphore(value: 0)
        var failure: Error?
        URLSession.shared.downloadTask(with: url) { tmp, resp, err in
            defer { sem.signal() }
            if let err = err { failure = err; return }
            guard let tmp = tmp, (resp as? HTTPURLResponse)?.statusCode == 200 else {
                failure = UpdateError(description: "Download failed (HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0))."); return
            }
            do { try FileManager.default.moveItem(at: tmp, to: dest) } catch { failure = error }
        }.resume()
        sem.wait()
        if let f = failure { throw f }
    }

    @discardableResult
    private func run(_ tool: String, _ args: [String]) throws -> String {
        let p = Process(); p.executableURL = URL(fileURLWithPath: tool); p.arguments = args
        let out = Pipe(); p.standardOutput = out; p.standardError = out
        try p.run(); p.waitUntilExit()
        let s = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        guard p.terminationStatus == 0 else { throw UpdateError(description: "\((tool as NSString).lastPathComponent) failed: \(s.prefix(200))") }
        return s
    }

    /// This app's Developer ID team (nil for ad-hoc / unsigned developer builds: those never self-install).
    static func ownTeamID() -> String? {
        var code: SecCode?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let c = code else { return nil }
        var st: SecStaticCode?
        guard SecCodeCopyStaticCode(c, [], &st) == errSecSuccess, let s = st else { return nil }
        return teamID(s)
    }

    private static func teamID(_ s: SecStaticCode) -> String? {
        var info: CFDictionary?
        guard SecCodeCopySigningInformation(s, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess,
              let d = info as? [String: Any] else { return nil }
        return d[kSecCodeInfoTeamIdentifier as String] as? String
    }

    /// The new app must be validly signed (deep, strict), by our team, with our bundle id.
    static func verify(_ app: URL) throws {
        guard let team = ownTeamID() else { throw UpdateError(description: "This build isn't Developer ID signed, so it can't update itself.") }
        var st: SecStaticCode?
        guard SecStaticCodeCreateWithPath(app as CFURL, [], &st) == errSecSuccess, let s = st else {
            throw UpdateError(description: "Couldn't read the new app's signature.")
        }
        let flags = SecCSFlags(rawValue: kSecCSCheckAllArchitectures | kSecCSCheckNestedCode | kSecCSStrictValidate)
        var req: SecRequirement?
        let reqText = "anchor apple generic and certificate leaf[subject.OU] = \"\(team)\" and identifier \"\(AppInfo.bundleID)\""
        guard SecRequirementCreateWithString(reqText as CFString, [], &req) == errSecSuccess else {
            throw UpdateError(description: "Couldn't build the signature check.")
        }
        let rc = SecStaticCodeCheckValidity(s, flags, req)
        guard rc == errSecSuccess else {
            throw UpdateError(description: "The new app's signature didn't pass (\(rc)): not signed by the same developer, or damaged.")
        }
    }

    /// Quit; a detached helper waits for this process to exit, moves the old app aside, puts the new one in its
    /// place, relaunches, and deletes the old copy only when the move worked (else it puts the old one back).
    private func swapAndRelaunch(newApp: URL, target: URL, work: URL, version: String) {
        let script = """
        pid=\(ProcessInfo.processInfo.processIdentifier)
        while kill -0 "$pid" 2>/dev/null; do sleep 0.3; done
        old="$1.old-$$"
        if mv "$1" "$old" && mv "$2" "$1"; then rm -rf "$old"; else [ -e "$1" ] || mv "$old" "$1"; fi
        xattr -dr com.apple.quarantine "$1" 2>/dev/null
        open "$1"
        rm -rf "$3"
        """
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", script, "onset-update", target.path, newApp.path, work.path]
        do { try p.run() } catch { appLog("update helper failed to start: \(error)"); return }
        appLog("update \(version): verified, relaunching")
        NSApp.terminate(nil)
    }
}
