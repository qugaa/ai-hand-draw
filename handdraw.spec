# PyInstaller build spec for AI Hand Draw.
#
#   pip install pyinstaller
#   pyinstaller handdraw.spec
#
# Produces dist/AI Hand Draw.exe - a single self-contained file.
# The hand-landmark model is NOT bundled; it is downloaded on first run into
# %LOCALAPPDATA%\AI Hand Draw\models, so the build stays smaller.

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# MediaPipe and CustomTkinter both ship data files (graphs, .tflite, .binarypb,
# themes, fonts) that a plain analysis misses, so pull everything they carry.
for pkg in ("mediapipe", "customtkinter"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# cv2 occasionally hides its extension modules behind a loader shim.
hiddenimports += collect_submodules("cv2")

block_cipher = None

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Nothing is excluded on purpose: mediapipe pulls in matplotlib, which
    # reaches pyparsing.testing and therefore needs the stdlib `unittest`.
    # Trimming those "obviously unused" modules breaks the app at startup.
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-file build: everything is packed into the EXE and unpacked to a temp
# folder at launch, so there is a single artefact to hand someone.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AI Hand Draw",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,          # no console window; it's a GUI app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico",
)
