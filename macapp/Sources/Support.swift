import Foundation
import Security

/// Bundle identity + the folders the app uses. Everything derives from Info.plist (written from macapp/app.conf).
enum AppInfo {
    static var name: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String ?? "Onset" }
    static var bundleID: String { Bundle.main.bundleIdentifier ?? "ai.whissle.onset" }
    static var version: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0" }
    static var resources: URL { Bundle.main.resourceURL! }
    static var python: URL { resources.appendingPathComponent("python/bin/python3") }
    static var appRoot: URL { resources.appendingPathComponent("app") }
    static var support: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent(name, isDirectory: true)
    }
    static var modelsDir: URL { support.appendingPathComponent("models", isDirectory: true) }
    static var dataDir: URL { support.appendingPathComponent("data", isDirectory: true) }
    static var logsDir: URL { support.appendingPathComponent("logs", isDirectory: true) }
    static var serverLog: URL { logsDir.appendingPathComponent("server.log") }
    /// Developer builds only (build_macapp.sh --dev-env): the repo's .env, read for keys missing from the Keychain.
    static var devEnvPath: String? {
        guard let p = Bundle.main.object(forInfoDictionaryKey: "HSDevEnvPath") as? String, !p.isEmpty else { return nil }
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }
}

/// API keys live in the login Keychain (generic passwords, service = bundle id). Values are never logged or written
/// to disk by the app; they reach the server only as environment variables of its process.
enum Keys {
    static let all = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY", "DEEPGRAM_API_KEY"]
    static let labels = ["ANTHROPIC_API_KEY": "Anthropic (answers)",
                         "TAVILY_API_KEY": "Tavily (web search; else Wikipedia)",
                         "DEEPGRAM_API_KEY": "Deepgram (cloud ear: Flux; else the local Arc ear)"]

    private static func query(_ account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: AppInfo.bundleID,
         kSecAttrAccount as String: account]
    }

    static func get(_ account: String) -> String? {
        var q = query(account)
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        let s = String(data: d, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (s?.isEmpty ?? true) ? nil : s
    }

    @discardableResult
    static func set(_ account: String, _ value: String?) -> Bool {
        let v = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        SecItemDelete(query(account) as CFDictionary)
        if v.isEmpty { return true }
        var q = query(account)
        q[kSecValueData as String] = Data(v.utf8)
        q[kSecAttrLabel as String] = "\(AppInfo.name): \(account)"
        q[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(q as CFDictionary, nil) == errSecSuccess
    }

    static func has(_ account: String) -> Bool { get(account) != nil }
}

/// KEY=VALUE lines of a .env file (developer fallback). Never logged.
func parseDotEnv(_ path: String) -> [String: String] {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [:] }
    var out: [String: String] = [:]
    for raw in text.split(whereSeparator: \.isNewline) {
        let line = raw.trimmingCharacters(in: .whitespaces)
        if line.isEmpty || line.hasPrefix("#") { continue }
        guard let eq = line.firstIndex(of: "=") else { continue }
        let k = line[..<eq].trimmingCharacters(in: .whitespaces)
        var v = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
        if v.count >= 2, let f = v.first, f == "\"" || f == "'", v.last == f { v = String(v.dropFirst().dropLast()) }
        out[k] = v
    }
    return out
}

/// A localhost TCP port nobody listens on. Prefers `preferred` (so the page origin, and with it the browser-side
/// memory id in localStorage, stays the same across launches), never 8770 (the demo server's port).
func freePort(preferred: Int?) -> Int {
    func bindable(_ port: Int) -> Int? {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        var one: Int32 = 1      // like uvicorn: a just-stopped server's TIME_WAIT socket must not block the same port
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(UInt16(port).bigEndian)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let r = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard r == 0 else { return nil }
        var got = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &got) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
        }
        return Int(UInt16(bigEndian: got.sin_port))
    }
    if let p = preferred, p != 8770, p > 1024, let ok = bindable(p) { return ok }
    for _ in 0..<20 {
        if let p = bindable(0), p != 8770 { return p }
    }
    return 8791
}
