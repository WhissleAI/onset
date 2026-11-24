# Build Onset for Windows (x64): dist\windows\Onset\ (the app folder) and dist\windows\Onset-<version>-win-x64-setup.exe.
# Run from the repo root in PowerShell. Full guide: docs\BUILD_WINDOWS.md.
#
#   powershell -ExecutionPolicy Bypass -File winapp\build_windows.ps1              # app folder + installer
#   powershell -ExecutionPolicy Bypass -File winapp\build_windows.ps1 -NoInstaller # app folder only (try it in place)
#   powershell -ExecutionPolicy Bypass -File winapp\build_windows.ps1 -Smoke       # + download models and run the smoke test
#
# Needs: uv (https://docs.astral.sh/uv/: irm https://astral.sh/uv/install.ps1 | iex), Inno Setup 6 (iscc.exe, for the
# installer; winget install JRSoftware.InnoSetup). No C compiler: vendor\arc\arc.dll and vendor\arctag\arctag.dll are
# committed (rebuilt on a Mac/Linux with scripts/build_windows_libs.sh). Optional signing: $env:ONSET_WIN_CERT (a .pfx)
# + $env:ONSET_WIN_CERT_PASSWORD, with signtool.exe on PATH (Windows SDK).
param([switch]$NoInstaller, [switch]$Smoke, [switch]$Clean)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$Root = Get-Location

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Conf($k) {
  $line = Get-Content macapp\app.conf | Where-Object { $_ -match "^$k=" } | Select-Object -First 1
  return ($line -replace "^$k=`"?([^`"#]*)`"?.*", '$1').Trim()
}
$AppName = Conf "APP_NAME"; $Version = Conf "APP_VERSION"; $PyVer = Conf "PYTHON_VERSION"
$Out = Join-Path $Root "dist\windows"; $App = Join-Path $Out $AppName; $Py = Join-Path $App "python"
Write-Host "$AppName $Version (Python $PyVer) -> $App"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv not found: irm https://astral.sh/uv/install.ps1 | iex (then open a new terminal)" }
foreach ($f in "vendor\arc\arc.dll", "vendor\arctag\arctag.dll") { if (-not (Test-Path $f)) { throw "$f missing (git pull; it is committed)" } }

Step "Python runtime (python-build-standalone $PyVer via uv)"
$stamp = (Get-FileHash macapp\requirements-native.txt).Hash + (Get-FileHash winapp\requirements-win.txt).Hash + $PyVer
$stampFile = Join-Path $Py ".onset-stamp"
if ($Clean -or -not (Test-Path $stampFile) -or (Get-Content $stampFile) -ne $stamp) {
  if (Test-Path $App) { Remove-Item -Recurse -Force $App }
  uv python install $PyVer
  $src = Split-Path -Parent (uv python find $PyVer)
  New-Item -ItemType Directory -Force $App | Out-Null
  Copy-Item -Recurse $src $Py
  Get-ChildItem $Py -Recurse -Filter EXTERNALLY-MANAGED | Remove-Item -Force   # our private copy: installs allowed
  uv pip install --python "$Py\python.exe" --link-mode copy --compile-bytecode -q -r macapp\requirements-native.txt -r winapp\requirements-win.txt
  Set-Content $stampFile $stamp
} else { Write-Host "  reusing $Py (requirements unchanged)" }
& "$Py\python.exe" -I -c "import sys, onnxruntime, fastapi, webview, keyring, clr; print('  python', sys.version.split()[0], 'onnxruntime', onnxruntime.__version__)"
if ($LASTEXITCODE) { throw "the runtime is missing a package" }

Step "App files"
foreach ($d in "server", "web", "scripts", "models", "macapp", "vendor\arc", "vendor\arctag", "eval\multipart", "index", "runs", "winapp", "Licenses") {
  New-Item -ItemType Directory -Force (Join-Path $App $d) | Out-Null
}
foreach ($d in "server", "web") {
  if (Test-Path "$App\$d") { Remove-Item -Recurse -Force "$App\$d" }
  Copy-Item -Recurse $d "$App\$d"
  Get-ChildItem "$App\$d" -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
}
Copy-Item scripts\fetch_models.py, scripts\smoke_typed.py "$App\scripts\"
$man = "models\manifest.json"; if (Test-Path models\manifest.release.json) { $man = "models\manifest.release.json" }
Copy-Item $man "$App\models\manifest.json"                         # the public first-run download (Dropbox, sha256)
Copy-Item macapp\app.conf, macapp\server_main.py "$App\macapp\"
Copy-Item vendor\arc\arc.dll, vendor\arc\LICENSE, vendor\arc\NOTICE "$App\vendor\arc\"
Copy-Item vendor\arctag\arctag.dll, vendor\arctag\LICENSE, vendor\arctag\NOTICE "$App\vendor\arctag\"
Copy-Item eval\questions.jsonl, eval\conversations.jsonl "$App\eval\"
Copy-Item eval\multipart\questions.jsonl "$App\eval\multipart\"
Copy-Item index\questions_local.jsonl "$App\index\"
Copy-Item winapp\onset_win.py, winapp\Onset.ico "$App\winapp\"
Copy-Item LICENSE "$App\Licenses\Onset-LICENSE.txt"; Copy-Item NOTICE "$App\Licenses\Onset-NOTICE.txt"
Copy-Item macapp\licenses\* "$App\Licenses\"
Copy-Item vendor\arctag\third_party\cjson\LICENSE "$App\Licenses\cJSON-LICENSE.txt"
& "$Py\python.exe" -I -m compileall -q "$App\server" "$App\macapp" "$App\scripts" "$App\winapp" | Out-Null

Step "Checks"
& "$Py\python.exe" -I -B -c "import ctypes, sys; [ctypes.CDLL(p) for p in sys.argv[1:]]; print('  arc.dll + arctag.dll load')" "$App\vendor\arc\arc.dll" "$App\vendor\arctag\arctag.dll"
if ($LASTEXITCODE) { throw "a native library did not load" }
Push-Location $App
& "$Py\python.exe" -I -B scripts\fetch_models.py status | Select-Object -First 12
Pop-Location

if ($Smoke) {
  Step "Smoke: models (one-time download to %LOCALAPPDATA%\$AppName\models) + a typed conversation"
  $env:MODELS_DIR = "$env:LOCALAPPDATA\$AppName\models"
  $env:KNOWLEDGE_DIR = "$env:MODELS_DIR\index-data\knowledge"
  $env:HEADSTART_DATA_DIR = "$Out\smoke-data"; $env:MEM_DIR = "$Out\smoke-data\memory"
  Push-Location $App
  & "$Py\python.exe" -I -B scripts\fetch_models.py --dest $env:MODELS_DIR fetch
  & "$Py\python.exe" -I -B scripts\smoke_typed.py --start
  $rc = $LASTEXITCODE
  Pop-Location
  if ($rc) { throw "smoke test failed (logs above)" }
}

function Sign($f) {
  if (-not $env:ONSET_WIN_CERT) { return }
  signtool sign /f $env:ONSET_WIN_CERT /p $env:ONSET_WIN_CERT_PASSWORD /fd sha256 /tr http://timestamp.digicert.com /td sha256 $f
  if ($LASTEXITCODE) { throw "signing $f failed" }
}
foreach ($exe in "python.exe", "pythonw.exe") { Sign "$Py\$exe" }

if (-not $NoInstaller) {
  Step "Installer (Inno Setup)"
  $iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
  if (-not $iscc) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
  if (-not (Test-Path $iscc)) { throw "Inno Setup 6 not found: winget install JRSoftware.InnoSetup (or -NoInstaller)" }
  & $iscc /Q "/DAppName=$AppName" "/DAppVersion=$Version" "/DSourceDir=$App" "/DOutDir=$Out" winapp\onset.iss
  if ($LASTEXITCODE) { throw "iscc failed" }
  $setup = Join-Path $Out "$AppName-$Version-win-x64-setup.exe"
  Sign $setup
  $sha = (Get-FileHash -Algorithm SHA256 $setup).Hash.ToLower()
  Set-Content "$setup.sha256" "$sha  $(Split-Path -Leaf $setup)"
  Write-Host "`n$setup`nsha256 $sha"
}
Write-Host "`nTry it in place: $Py\pythonw.exe -I -B $App\winapp\onset_win.py"
