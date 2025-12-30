# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# Project root
# Note: When PyInstaller executes the spec file, `__file__` may not be defined
# in some environments. PyInstaller provides `SPECPATH` for this purpose.
_spec_dir = Path(globals().get("SPECPATH") or os.getcwd()).resolve()
ROOT = _spec_dir

block_cipher = None

# Collect package data/binaries/hidden imports for ML stack.
# NOTE: PyTorch/Ultralytics makes the build very large.
_datas: list[tuple[str, str]] = []
_binaries: list[tuple[str, str]] = []
_hiddenimports: list[str] = []

for pkg in [
    "customtkinter",
    "cv2",
    "numpy",
    "PIL",
    "ultralytics",
    "torch",
    "torchvision",
]:
    try:
        d, b, h = collect_all(pkg)
        _datas += d
        _binaries += b
        _hiddenimports += h
    except Exception:
        # Some optional deps may not exist in the environment; ignore.
        pass

# Optional YAML support
for pkg in ["yaml"]:
    try:
        d, b, h = collect_all(pkg)
        _datas += d
        _binaries += b
        _hiddenimports += h
    except Exception:
        pass

# Include runtime assets/models if present
# - weights/* (classifier + optional detector)
# - yolov8n.pt (default COCO detector)
# - datasets/data.yaml (class-name mapping)
# - assets/fonts/* (Vietnamese font)

def add_if_exists(src: Path, dst_rel: str):
    if src.exists():
        _datas.append((str(src), dst_rel))


def add_tree_if_exists(src_dir: Path, dst_rel: str):
    if not src_dir.exists() or not src_dir.is_dir():
        return
    for p in src_dir.rglob("*"):
        if p.is_file():
            rel_parent = str(Path(dst_rel) / p.relative_to(src_dir).parent)
            _datas.append((str(p), rel_parent))


add_tree_if_exists(ROOT / "weights", "weights")
add_if_exists(ROOT / "yolov8n.pt", ".")
add_if_exists(ROOT / "yolo11n.pt", ".")
add_if_exists(ROOT / "yolov8n-cls.pt", ".")
add_if_exists(ROOT / "yolov8n.pt", ".")
add_if_exists(ROOT / "datasets" / "data.yaml", os.path.join("datasets"))
add_if_exists(ROOT / "datasets" / "classifier_data.yaml", os.path.join("datasets"))
add_tree_if_exists(ROOT / "assets" / "fonts", os.path.join("assets", "fonts"))


a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="BananaQualityGrading",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
