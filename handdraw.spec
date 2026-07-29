# PyInstaller build spec for AI Hand Draw.
#
#   pip install pyinstaller
#   pyinstaller handdraw.spec
#
# Produces dist/AI Hand Draw/AI Hand Draw.exe (a self-contained folder).
# The hand-landmark model is NOT bundled; it is downloaded on first run into
# %LOCALAPPDATA%\AI Hand Draw\models, so the build stays small.

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
    excludes=["tkinter.test", "test", "unittest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI Hand Draw",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no console window; it's a GUI app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AI Hand Draw",
)
