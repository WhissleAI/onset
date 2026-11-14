import Foundation

/// Runs the bundled Python server on a free localhost port, waits for /api/health warm, restarts it if it crashes,
/// stops it on quit. Also runs scripts/fetch_models.py (model status / download / import) with the same Python.
final class ServerManager {
    enum State: Equatable {
        case idle
        case checkingModels
        case needsModels(missing: [ModelRow])
        case downloading(done: Int64, total: Int64, file: String)
        case starting(seconds: Int, detail: String)
        case ready(port: Int)
        case failed(String)
    }

    struct ModelRow: Equatable, Decodable {
        let model: String
        let name: String
        let required: Bool
        let ok: Bool
        let bytes: Int64
        let downloadable: Bool
    }

    var onState: ((State) -> Void)?
    private(set) var state: State = .idle { didSet { let s = state; DispatchQueue.main.async { self.onState?(s) } } }
    private(set) var port = 0
    private(set) var generation = 0          // +1 per server launch: the page reloads against each new server
    private var proc: Process?
    private var stopping = false
    private var crashes: [Date] = []
    private var pollTimer: DispatchSourceTimer?
    private var startedAt = Date()
    private let q = DispatchQueue(label: "server-manager")

    // MARK: environment

    /// The server's environment: a clean copy of ours, model + data folders, keys (Keychain, else the dev .env), ear.
    func serverEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        for k in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONSTARTUP"] { env.removeValue(forKey: k) }
        for k in Keys.all { env.removeValue(forKey: k) }
        var dev: [String: String] = [:]
        if let p = AppInfo.devEnvPath { dev = parseDotEnv(p) }
        // non-secret settings from the dev .env first (EAR, FLUX_SOFT, DIRECTOR, ...), then ours on top
        for (k, v) in dev where !Keys.all.contains(k) { env[k] = v }
        for k in Keys.all {
            if let v = Keys.get(k) { env[k] = v } else if let v = dev[k], !v.isEmpty { env[k] = v }
        }
        env["APP_NAME"] = AppInfo.name
        env["MODELS_DIR"] = AppInfo.modelsDir.path
        env["HEADSTART_DATA_DIR"] = AppInfo.dataDir.path
        env["HF_HUB_OFFLINE"] = "1"                 // models come from MODELS_DIR, never a surprise download
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"        // the bundle is signed: never write into it
        env["TOKENIZERS_PARALLELISM"] = "false"
        // reuse earlier searches across launches: the semantic search cache persists to data/search_cache.json
        // (bounded by CACHE_MAX, atomic writes, TTLs live 10 min / general 6 h / stable 3 days; hits labelled "cached")
        if env["CACHE_PERSIST"] == nil { env["CACHE_PERSIST"] = "1" }
        env["CACHE_PATH"] = AppInfo.dataDir.appendingPathComponent("search_cache.json").path
        env["MEM_DIR"] = AppInfo.dataDir.appendingPathComponent("memory").path
        let ear = UserDefaults.standard.string(forKey: "ear") ?? "auto"
        switch ear {
        case "arc": env["EAR"] = "arc"
        case "flux": env["EAR"] = "flux"
        default:  // auto: the cloud ear with a Deepgram key, else the local Arc ear
            if env["EAR"] == nil || dev["EAR"] != nil { env["EAR"] = env["DEEPGRAM_API_KEY"] != nil ? (dev["EAR"] ?? "flux") : "arc" }
        }
        return env
    }

    // MARK: models

    private func runPython(_ args: [String], onLine: ((String) -> Void)? = nil) -> (Int32, String) {
        let p = Process()
        p.executableURL = AppInfo.python
        p.arguments = ["-I", AppInfo.appRoot.appendingPathComponent("scripts/fetch_models.py").path,
                       "--dest", AppInfo.modelsDir.path] + args
        p.environment = serverEnvironment()
        p.currentDirectoryURL = AppInfo.appRoot
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        var all = Data()
        var buf = Data()
        let lock = NSLock()
        pipe.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty { return }
            lock.lock(); all.append(d); buf.append(d)
            while let nl = buf.firstIndex(of: 0x0A) {
                let line = String(data: buf[buf.startIndex..<nl], encoding: .utf8) ?? ""
                buf.removeSubrange(buf.startIndex...nl)
                onLine?(line)
            }
            lock.unlock()
        }
        do { try p.run() } catch { return (-1, "\(error)") }
        p.waitUntilExit()
        pipe.fileHandleForReading.readabilityHandler = nil
        all.append(pipe.fileHandleForReading.readDataToEndOfFile())
        return (p.terminationStatus, String(data: all, encoding: .utf8) ?? "")
    }

    func modelStatus() -> [ModelRow] {
        let (_, out) = runPython(["status", "--json"])
        struct S: Decodable { let models: [ModelRow] }
        guard let line = out.split(separator: "\n").last(where: { $0.hasPrefix("{") }),
              let s = try? JSONDecoder().decode(S.self, from: Data(line.utf8)) else { return [] }
        return s.models
    }

    /// Check models; start the server if every required one is there, else report what is missing.
    func start() {
        q.async {
            self.state = .checkingModels
            try? FileManager.default.createDirectory(at: AppInfo.modelsDir, withIntermediateDirectories: true)
            let rows = self.modelStatus()
            let missing = rows.filter { $0.required && !$0.ok }
            if rows.isEmpty {
                self.state = .failed("Could not read the model manifest (see Open Logs).")
            } else if !missing.isEmpty {
                self.state = .needsModels(missing: missing)
            } else {
                self.launch()
            }
        }
    }

    func downloadModels() {
        q.async {
            self.state = .downloading(done: 0, total: 0, file: "")
            _ = self.runPython(["--json-progress", "fetch"]) { line in
                guard let d = line.data(using: .utf8),
                      let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                      (o["event"] as? String) == "progress" else { return }
                let done = (o["done"] as? NSNumber)?.int64Value ?? 0
                let total = (o["total"] as? NSNumber)?.int64Value ?? 0
                self.state = .downloading(done: done, total: total, file: "\(o["model"] ?? "")/\(o["file"] ?? "")")
            }
            self.q.async { self.start() }
        }
    }

    /// Copy local-only models in from a folder (the folder itself, or a folder holding <dir>/ subfolders).
    func importModels(from folder: URL, rows: [ModelRow]) -> String {
        var specs: [String] = []
        for r in rows where !r.downloadable {
            let sub = folder.appendingPathComponent(r.model)
            let src = FileManager.default.fileExists(atPath: sub.path) ? sub : folder
            specs.append("\(r.model)=\(src.path)")
        }
        guard !specs.isEmpty else { return "" }
        let (_, out) = runPython(["import"] + specs)
        q.async { self.start() }
        return out
    }

    // MARK: server process

    private func launch() {
        stopping = false
        let fm = FileManager.default
        try? fm.createDirectory(at: AppInfo.logsDir, withIntermediateDirectories: true)
        try? fm.createDirectory(at: AppInfo.dataDir, withIntermediateDirectories: true)
        port = freePort(preferred: UserDefaults.standard.object(forKey: AppInfo.portKey) as? Int)
        UserDefaults.standard.set(port, forKey: AppInfo.portKey)
        generation += 1
        // keep the last 5 logs: server.log (this run), server.log.1 (previous) ... server.log.5
        let logURL = { (i: Int) in AppInfo.logsDir.appendingPathComponent(i == 0 ? "server.log" : "server.log.\(i)") }
        try? fm.removeItem(at: logURL(5))
        for i in stride(from: 4, through: 0, by: -1) { try? fm.moveItem(at: logURL(i), to: logURL(i + 1)) }
        fm.createFile(atPath: AppInfo.serverLog.path, contents: nil)
        let log = try? FileHandle(forWritingTo: AppInfo.serverLog)

        let p = Process()
        p.executableURL = AppInfo.python
        p.arguments = ["-I", AppInfo.appRoot.appendingPathComponent("macapp/server_main.py").path,
                       "--port", String(port), "--parent-pid", String(getpid())]
        p.environment = serverEnvironment()
        p.currentDirectoryURL = AppInfo.appRoot
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { [weak self] pr in self?.q.async { self?.exited(pr) } }
        do {
            try p.run()
        } catch {
            state = .failed("Could not start the bundled Python: \(error.localizedDescription)")
            return
        }
        proc = p
        startedAt = Date()
        state = .starting(seconds: 0, detail: "Starting the local server")
        poll()
    }

    private func poll() {
        pollTimer?.cancel()
        let t = DispatchSource.makeTimerSource(queue: q)
        t.schedule(deadline: .now() + 0.5, repeating: 0.5)
        t.setEventHandler { [weak self] in self?.checkHealth() }
        pollTimer = t
        t.resume()
    }

    private var inFlight = false
    private func checkHealth() {
        guard let p = proc, p.isRunning, !inFlight else { return }
        let secs = Int(Date().timeIntervalSince(startedAt))
        if secs > 300 {
            pollTimer?.cancel()
            state = .failed("The server did not warm up within 5 minutes (see Open Logs).")
            return
        }
        inFlight = true
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/health")!)
        req.timeoutInterval = 2
        URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
            guard let self = self else { return }
            self.q.async {
                self.inFlight = false
                if let d = data, let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                   (o["warm"] as? Bool) == true {
                    self.pollTimer?.cancel()
                    self.state = .ready(port: self.port)
                } else {
                    let detail = data == nil ? "Starting the local server" : "Warming up the models (Arc, Arc Tagger, EmbeddingGemma 2)"
                    if case .ready = self.state { return }
                    self.state = .starting(seconds: secs, detail: detail)
                }
            }
        }.resume()
    }

    private func exited(_ p: Process) {
        guard p === proc else { return }
        proc = nil
        pollTimer?.cancel()
        if stopping { state = .idle; return }
        let now = Date()
        crashes = crashes.filter { now.timeIntervalSince($0) < 120 } + [now]
        if crashes.count > 3 {
            state = .failed("The server stopped \(crashes.count) times in 2 minutes (exit \(p.terminationStatus)). See Open Logs.")
            return
        }
        state = .starting(seconds: 0, detail: "The server stopped (exit \(p.terminationStatus)); restarting")
        q.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self = self, !self.stopping, self.proc == nil else { return }
            self.launch()
        }
    }

    /// SIGTERM, then SIGKILL after `timeout`. Blocks the caller until the process is gone.
    func stop(timeout: TimeInterval = 4.0) {
        let p: Process? = q.sync {
            stopping = true
            pollTimer?.cancel()
            return proc
        }
        guard let pr = p, pr.isRunning else { return }
        pr.terminate()
        let deadline = Date().addingTimeInterval(timeout)
        while pr.isRunning && Date() < deadline { usleep(50_000) }
        if pr.isRunning { kill(pr.processIdentifier, SIGKILL) }
        pr.waitUntilExit()
    }

    func restart() {
        DispatchQueue.global().async {
            self.stop()
            self.q.async { self.crashes = []; self.start() }
        }
    }
}
