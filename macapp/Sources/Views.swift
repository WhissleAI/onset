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
            Text("Ask out loud. The answer is ready when you stop.").foregroundColor(.secondary)
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
            Text("Starting…").foregroundColor(.secondary)
        case .starting(let s, let detail):
            ProgressView().progressViewStyle(.linear)
            Text("Getting ready… \(s) s").foregroundColor(.secondary).font(.callout)
            Text("The first start takes about a minute.").font(.caption).foregroundColor(.secondary)
        case .downloading(let done, let total, let file):
            ProgressView(value: total > 0 ? Double(done) / Double(total) : 0).progressViewStyle(.linear)
            Text(total > 0 ? "Downloading: \(done / 1_000_000) of \(total / 1_000_000) MB" : "Preparing download…")
                .font(.callout)
            Text(file).font(.caption).foregroundColor(.secondary).lineLimit(1).truncationMode(.middle)
        case .needsModels(let missing) where missing.allSatisfy({ $0.downloadable }):
            // first run: one download, no choices to make
            let mb = missing.reduce(Int64(0)) { $0 + $1.bytes } / 1_000_000
            VStack(spacing: 8) {
                Text("One-time setup").font(.headline)
                Text("\(AppInfo.name) listens and answers on this Mac, so your voice never leaves it. It needs a one-time download (\(mb >= 1000 ? String(format: "%.1f GB", Double(mb) / 1000) : "\(mb) MB")).")
                    .multilineTextAlignment(.center).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Download", action: onDownload).keyboardShortcut(.defaultAction).controlSize(.large)
                Button("Check Again", action: onRetry)
            }
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
    @State private var collect: Bool = UserDefaults.standard.object(forKey: "trainCollect") as? Bool ?? true
    @State private var note = ""
    var onSave: () -> Void

    @State private var advanced = false

    var body: some View {
        Form {
            Section(header: Text("Privacy").font(.headline)) {
                Toggle("Help improve Onset", isOn: $collect)
                Text("Keeps anonymous timings of your questions on this Mac to tune Onset. No audio. Nothing is uploaded.")
                    .font(.caption).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            DisclosureGroup(isExpanded: $advanced) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Onset works without any keys: answers are written on this Mac and web results come from DuckDuckGo. Add your own keys for longer answers or other services. Keys stay in your Keychain.")
                        .font(.caption).foregroundColor(.secondary).fixedSize(horizontal: false, vertical: true)
                        .padding(.bottom, 4)
                    ForEach(Keys.all, id: \.self) { k in
                        KeyRow(info: KeyInfo.of(k),
                               entry: Binding(get: { entries[k] ?? "" }, set: { entries[k] = $0 }),
                               saved: saved[k] == true, compact: true,
                               onRemove: { Keys.set(k, nil); refresh(); note = "Removed. Save to apply." })
                            .padding(.bottom, 4)
                    }
                    Text("Speech recognition").font(.subheadline).padding(.top, 6)
                    Picker("", selection: $ear) {
                        Text("On this Mac (recommended)").tag("auto")
                        Text("On this Mac only").tag("arc")
                        Text("Deepgram (cloud, needs a key)").tag("flux")
                    }.pickerStyle(.radioGroup).labelsHidden()
                }.padding(.top, 6)
            } label: { Text("Advanced").font(.headline) }
            .padding(.top, 10)
            HStack {
                Text(note).font(.caption).foregroundColor(.secondary)
                Spacer()
                Button("Save") {
                    for (k, v) in entries where !v.trimmingCharacters(in: .whitespaces).isEmpty { Keys.set(k, v) }
                    entries = [:]
                    UserDefaults.standard.set(ear, forKey: "ear")
                    UserDefaults.standard.set(collect, forKey: "trainCollect")
                    refresh()
                    note = "Saved."
                    onSave()
                }.keyboardShortcut(.defaultAction)
            }.padding(.top, 10)
        }
        .padding(20)
        .frame(width: 520)
        .onAppear(perform: refresh)
    }

    private func refresh() {
        var s: [String: Bool] = [:]
        for k in Keys.all { s[k] = Keys.has(k) }
        saved = s
    }
}
