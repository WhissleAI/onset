; Onset for Windows installer (Inno Setup 6). Built by winapp\build_windows.ps1, which passes:
;   /DAppName=Onset /DAppVersion=x.y.z /DSourceDir=dist\windows\Onset /DOutDir=dist\windows
; Per-user install (no admin prompt) into %LOCALAPPDATA%\Programs\Onset. Models, memory and logs live in
; %LOCALAPPDATA%\Onset (kept on uninstall unless the user ticks the box; an upgrade keeps them).

#ifndef AppName
  #define AppName "Onset"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6E7C1B0A-3F7D-4C2B-9B57-0A1C5D3E0B11}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Whissle
AppPublisherURL=https://github.com/WhissleAI/onset
AppSupportURL=https://github.com/WhissleAI/onset/issues
AppUpdatesURL=https://github.com/WhissleAI/onset/releases
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutDir}
OutputBaseFilename={#AppName}-{#AppVersion}-win-x64-setup
SetupIconFile={#SourceDir}\winapp\Onset.ico
UninstallDisplayIcon={app}\winapp\Onset.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; an upgrade replaces the code folders whole (no stale modules from the previous version)
Type: filesandordirs; Name: "{app}\server"
Type: filesandordirs; Name: "{app}\web"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-I -B ""{app}\winapp\onset_win.py"""; WorkingDir: "{app}"; IconFilename: "{app}\winapp\Onset.ico"; AppUserModelID: "ai.whissle.onset"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-I -B ""{app}\winapp\onset_win.py"""; WorkingDir: "{app}"; IconFilename: "{app}\winapp\Onset.ico"; AppUserModelID: "ai.whissle.onset"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-I -B ""{app}\winapp\onset_win.py"""; WorkingDir: "{app}"; Description: "Open {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; stop only Onset's own Python processes (their executable is inside the install folder)
Filename: "powershell.exe"; Parameters: "-NoProfile -Command ""Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '{app}\*' } | Stop-Process -Force"""; Flags: runhidden; RunOnceId: "StopOnset"

[Code]
var RemoveData: Boolean;

function WebView2Installed(): Boolean;
var v: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v) or
            RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v);
  if Result then Result := (v <> '') and (v <> '0.0.0.0');
end;

function InitializeSetup(): Boolean;
var code: Integer;
begin
  Result := True;
  if not WebView2Installed() then
  begin
    if MsgBox('Onset needs the Microsoft Edge WebView2 Runtime (built into Windows 11 and most Windows 10 PCs). ' +
              'Open the download page now? Install it, then run this setup again.', mbConfirmation, MB_YESNO) = IDYES then
      ShellExec('open', 'https://developer.microsoft.com/microsoft-edge/webview2/#download', '', '', SW_SHOW, ewNoWait, code);
    Result := False;
  end;
end;

function InitializeUninstall(): Boolean;
begin
  RemoveData := MsgBox('Also remove Onset''s downloaded models, memory and logs (%LOCALAPPDATA%\Onset)?',
                       mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and RemoveData then
    DelTree(ExpandConstant('{localappdata}\{#AppName}'), True, True, True);
end;
