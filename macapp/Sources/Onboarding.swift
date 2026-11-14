import AppKit
import SwiftUI

/// What each API key powers, where to get it, and what happens without it (first-run sheet + Settings).
struct KeyInfo {
    let env: String
    let title: String
    let powers: String
    let without: String
    let getURL: String
    let optional: Bool

    static let all: [KeyInfo] = [
        KeyInfo(env: "DEEPGRAM_API_KEY", title: "Deepgram",
                powers: "Hears you in the cloud with Deepgram Flux: the quickest turn-taking.",
                without: "Arc listens on this Mac instead. Private, a little slower.",
                getURL: "https://console.deepgram.com/signup", optional: false),
        KeyInfo(env: "TAVILY_API_KEY", title: "Tavily",
                powers: "Searches the live web: fresh, ranked results for news, sports, prices and more.",
                without: "Telem searches the web if you add its key. With neither, answers come from Wikipedia.",
                getURL: "https://app.tavily.com", optional: false),
        KeyInfo(env: "TELEM_API_KEY", title: "Telem",
                powers: "A web search router across providers. Used when there is no Tavily key, and as a backup when Tavily fails.",
                without: "Tavily alone, or Wikipedia when neither key is set.",
                getURL: "https://app.telem.ai", optional: false),
        KeyInfo(env: "ANTHROPIC_API_KEY", title: "Anthropic (optional)",
                powers: "Claude writes a short answer from the sources it found.",
                without: "The answer is the best sentence from the top source, shown with its link.",
                getURL: "https://console.anthropic.com/settings/keys", optional: true),
    ]
    static func of(_ env: String) -> KeyInfo { all.first { $0.env == env }! }
}

/// The services Onset will use with a given set of keys (the same rules ServerManager applies to the server).
struct ServicePlan {
    let ear: String
    let web: String
    let answers: String

    static func webBackend(tavily: Bool, telem: Bool) -> String { tavily ? "tavily" : (telem ? "telem" : "wikipedia") }

    static func with(_ has: (String) -> Bool) -> ServicePlan {
        let earPref = UserDefaults.standard.string(forKey: "ear") ?? "auto"
        let flux = has("DEEPGRAM_API_KEY") && earPref != "arc"
        let web: String
        switch webBackend(tavily: has("TAVILY_API_KEY"), telem: has("TELEM_API_KEY")) {
        case "tavily": web = has("TELEM_API_KEY") ? "Tavily, with Telem as backup" : "Tavily"
        case "telem": web = "Telem"
        default: web = "Wikipedia (free)"
        }
        return ServicePlan(ear: flux ? "Deepgram Flux" : "Arc, on this Mac",
                           web: web,
                           answers: has("ANTHROPIC_API_KEY") ? "Claude" : "Best source sentence")
    }
}

/// One tiny, free call per service that proves a key works. Never logs the key.
enum KeyCheck {
    enum Result: Equatable { case idle, checking, works(String), failed(String) }

    static func run(_ env: String, key: String, done: @escaping (Result) -> Void) {
        var req: URLRequest
        switch env {
        case "ANTHROPIC_API_KEY":       // list models: free
            req = URLRequest(url: URL(string: "https://api.anthropic.com/v1/models?limit=1")!)
            req.setValue(key, forHTTPHeaderField: "x-api-key")
            req.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        case "TAVILY_API_KEY":          // account usage: free, no search credit
            req = URLRequest(url: URL(string: "https://api.tavily.com/usage")!)
            req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        case "DEEPGRAM_API_KEY":        // list projects: free
            req = URLRequest(url: URL(string: "https://api.deepgram.com/v1/projects")!)
            req.setValue("Token \(key)", forHTTPHeaderField: "Authorization")
        case "TELEM_API_KEY":           // an empty search: the key is checked first (401), then the body (400): never billed
            req = URLRequest(url: URL(string: "https://router.telem.ai/v1/search")!)
            req.httpMethod = "POST"
            req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = Data("{}".utf8)
        default:
            return done(.failed("Unknown key"))
        }
        req.timeoutInterval = 10
        let name = KeyInfo.of(env).title.replacingOccurrences(of: " (optional)", with: "")
        URLSession.shared.dataTask(with: req) { _, resp, err in
            let r: Result
            if let e = err {
                r = .failed("Couldn't reach \(name): \(e.localizedDescription)")
            } else {
                let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                switch code {
                case 200..<300: r = .works("Works")
                case 400, 422: r = env == "TELEM_API_KEY" ? .works("Works") : .failed("\(name) answered HTTP \(code).")
                case 401, 403: r = .failed("\(name) didn't accept this key. Check that you copied all of it.")
                case 402: r = .failed("The key is valid but the \(name) account is out of credit.")
                case 429: r = .failed("\(name) says too many requests. Try again in a minute.")
                case 500...: r = .failed("\(name) is having trouble right now (HTTP \(code)). Try again later.")
                default: r = .failed("\(name) answered HTTP \(code).")
                }
            }
            DispatchQueue.main.async { done(r) }
        }.resume()
    }
}

/// One key: what it does, a secure field, Test, Remove, and where to get one. Used by onboarding and Settings.
struct KeyRow: View {
    let info: KeyInfo
    @Binding var entry: String
    let saved: Bool
    var compact = false
    var onRemove: (() -> Void)? = nil
    @State private var check: KeyCheck.Result = .idle

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .firstTextBaseline) {
                Text(info.title).font(.headline)
                Spacer()
                Link("Get a key", destination: URL(string: info.getURL)!).font(.callout)
            }
            if !compact {
                Text(info.powers).font(.callout).fixedSize(horizontal: false, vertical: true)
                Text("Without it: \(info.without)").font(.caption).foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                SecureField(saved ? "Saved in Keychain (type to replace)" : "Paste your \(info.title.replacingOccurrences(of: " (optional)", with: "")) key",
                            text: $entry)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: entry) { _ in check = .idle }
                Button("Test") {
                    let typed = entry.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard let k = typed.isEmpty ? Keys.get(info.env) : typed else {
                        check = .failed("Paste a key first.")
                        return
                    }
                    check = .checking
                    KeyCheck.run(info.env, key: k) { check = $0 }
                }
                .disabled(entry.trimmingCharacters(in: .whitespaces).isEmpty && !saved)
                if saved, let rm = onRemove { Button("Remove", action: rm) }
            }
            switch check {
            case .idle: EmptyView()
            case .checking: Text("Checking…").font(.caption).foregroundColor(.secondary)
            case .works(let m): Label(m, systemImage: "checkmark.circle.fill").font(.caption).foregroundColor(.green)
            case .failed(let m): Label(m, systemImage: "exclamationmark.triangle.fill").font(.caption).foregroundColor(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// First-run sheet: Onset works with no keys; each key makes one part better. Keys go only to the Keychain.
struct OnboardingView: View {
    @State private var entries: [String: String] = [:]
    @State private var saved: [String: Bool] = [:]
    var onDone: (_ changed: Bool) -> Void

    private func has(_ k: String) -> Bool {
        !(entries[k] ?? "").trimmingCharacters(in: .whitespaces).isEmpty || saved[k] == true
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Welcome to \(AppInfo.name)").font(.system(size: 22, weight: .semibold))
                Text("Ask out loud or type, and \(AppInfo.name) searches while you're still asking. It works right away with no keys: Arc listens on this Mac and answers come from Wikipedia. Add your own keys to make each part better.")
                    .foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    ForEach(KeyInfo.all, id: \.env) { info in
                        KeyRow(info: info,
                               entry: Binding(get: { entries[info.env] ?? "" }, set: { entries[info.env] = $0 }),
                               saved: saved[info.env] == true,
                               onRemove: { Keys.set(info.env, nil); refresh() })
                        Divider()
                    }
                }.padding(.trailing, 8)
            }
            let plan = ServicePlan.with(has)
            VStack(alignment: .leading, spacing: 2) {
                Text("With these keys").font(.caption).foregroundColor(.secondary)
                Text("Ear: \(plan.ear)  ·  Web: \(plan.web)  ·  Answers: \(plan.answers)").font(.callout)
            }
            Text("Keys are stored only in your Mac's Keychain and sent only to their own service. Your questions, memory and history stay on this Mac. Change keys any time in Settings (⌘,).")
                .font(.caption).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Skip: use the free mode") { finish(save: false) }
                Spacer()
                Button("Continue") { finish(save: true) }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 620, height: 720)
        .background(Color(NSColor.windowBackgroundColor))
        .onAppear(perform: refresh)
    }

    private func refresh() {
        var s: [String: Bool] = [:]
        for k in Keys.all { s[k] = Keys.has(k) }
        saved = s
    }

    private func finish(save: Bool) {
        var changed = false
        if save {
            for (k, v) in entries where !v.trimmingCharacters(in: .whitespaces).isEmpty { Keys.set(k, v); changed = true }
        }
        entries = [:]
        AppInfo.onboardingDone = true
        onDone(changed)
    }
}
