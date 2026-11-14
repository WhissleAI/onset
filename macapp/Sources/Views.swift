import AppKit
import SwiftUI

/// Observable mirror of ServerManager.State for the SwiftUI splash.
final class SplashModel: ObservableObject {
    @Published var state: ServerManager.State = .idle
}

struct SplashView: View {
    @ObservedObject var model: SplashModel
    var onDownload: () -> Void
    var onImport: () -> Void
    var onOpenLogs: () -> Void
    var onRetry: () -> Void
    var onSettings: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 96, height: 96)
            Text(AppInfo.name).font(.system(size: 30, weight: .semibold))
            Text("Voice search that searches while you speak").foregroundColor(.secondary)
            content.frame(maxWidth: 460)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(NSColor.windowBackgroundColor))
    }

    @ViewBuilder var content: some View {
        switch model.state {
        case .idle, .checkingModels:
            ProgressView().controlSize(.small)
            Text("Checking models…").foregroundColor(.secondary)
        case .starting(let s, let detail):
            ProgressView().progressViewStyle(.linear)
            Text("\(detail)… \(s) s").foregroundColor(.secondary).font(.callout)
            Text("First start takes about a minute: the models load into memory.").font(.caption).foregroundColor(.secondary)
        case .downloading(let done, let total, let file):
            ProgressView(value: total > 0 ? Double(done) / Double(total) : 0).progressViewStyle(.linear)
            Text(total > 0 ? "Downloading models: \(done / 1_000_000) of \(total / 1_000_000) MB" : "Preparing download…")
                .font(.callout)
            Text(file).font(.caption).foregroundColor(.secondary).lineLimit(1).truncationMode(.middle)
        case .needsModels(let missing):
            VStack(alignment: .leading, spacing: 6) {
                Text("Models needed").font(.headline)
                ForEach(missing, id: \.model) { r in
                    HStack {
                        Text(r.model).font(.system(.body, design: .monospaced))
                        Spacer()
                        Text("\(r.bytes / 1_000_000) MB").foregroundColor(.secondary)
                        Text(r.downloadable ? "download" : "import").foregroundColor(r.downloadable ? .accentColor : .orange)
                    }
                }
                Text("Models go to ~/Library/Application Support/\(AppInfo.name)/models. Models marked import have no public download: choose a folder that holds them.")
                    .font(.caption).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                if missing.contains(where: { $0.downloadable }) { Button("Download", action: onDownload).keyboardShortcut(.defaultAction) }
                if missing.contains(where: { !$0.downloadable }) { Button("Import Folder…", action: onImport) }
                Button("Check Again", action: onRetry)
            }
        case .ready:
            ProgressView().controlSize(.small)
            Text("Loading…").foregroundColor(.secondary)
        case .failed(let msg):
            Text(msg).foregroundColor(.red).multilineTextAlignment(.center).fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Try Again", action: onRetry).keyboardShortcut(.defaultAction)
                Button("Open Logs", action: onOpenLogs)
                Button("Settings…", action: onSettings)
            }
        }
    }
}

/// Settings: API keys (Keychain) + ear. Saved keys are never shown, only "saved".
struct SettingsView: View {
    @State private var entries: [String: String] = [:]
    @State private var saved: [String: Bool] = [:]
    @State private var ear: String = UserDefaults.standard.string(forKey: "ear") ?? "auto"
    @State private var note = ""
    var onSave: () -> Void

    var body: some View {
        Form {
            Section(header: Text("API keys").font(.headline)) {
                ForEach(Keys.all, id: \.self) { k in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(Keys.labels[k] ?? k).font(.callout)
                        HStack {
                            SecureField(saved[k] == true ? "saved in Keychain (type to replace)" : "not set",
                                        text: Binding(get: { entries[k] ?? "" }, set: { entries[k] = $0 }))
                                .textFieldStyle(.roundedBorder)
                                .frame(minWidth: 320)
                            if saved[k] == true {
                                Button("Remove") { Keys.set(k, nil); refresh(); note = "Removed \(k). Save to restart the server." }
                            }
                        }
                    }
                }
                Text("Stored in your login Keychain and passed to the local server as environment variables. Never written to disk or logs.")
                    .font(.caption).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Section(header: Text("Ear").font(.headline).padding(.top, 8)) {
                Picker("", selection: $ear) {
                    Text("Automatic (Deepgram Flux with a key, else Arc on this Mac)").tag("auto")
                    Text("Arc (on this Mac, no key)").tag("arc")
                    Text("Deepgram Flux (cloud; falls back to Arc)").tag("flux")
                }.pickerStyle(.radioGroup).labelsHidden()
            }
            HStack {
                Text(note).font(.caption).foregroundColor(.secondary)
                Spacer()
                Button("Save and Restart Server") {
                    for (k, v) in entries where !v.trimmingCharacters(in: .whitespaces).isEmpty { Keys.set(k, v) }
                    entries = [:]
                    UserDefaults.standard.set(ear, forKey: "ear")
                    refresh()
                    note = "Saved."
                    onSave()
                }.keyboardShortcut(.defaultAction)
            }.padding(.top, 10)
        }
        .padding(20)
        .frame(width: 540)
        .onAppear(perform: refresh)
    }

    private func refresh() {
        var s: [String: Bool] = [:]
        for k in Keys.all { s[k] = Keys.has(k) }
        saved = s
    }
}
