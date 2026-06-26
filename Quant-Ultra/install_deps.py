#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quant-Ultra + Conformal-BL 项目依赖自动检查与安装脚本
用法: python install_deps.py [--mirror https://pypi.tuna.tsinghua.edu.cn/simple] [--scan]
"""

import subprocess
import sys
import importlib
import re
import os
import argparse
from pathlib import Path

# 项目所需的核心第三方库（根据代码实际导入情况整理）
REQUIRED_LIBRARIES = [
    "numpy",
    "pandas",
    "pytz",
    "lightgbm",
    "scikit-learn",      # 导入为 sklearn
    "statsmodels",
    "scipy",
    "cvxpy",
    "matplotlib",        # 部分可视化可能用到
    "tabulate",          # 常见辅助
]

# 标准库白名单（不安装）
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set([
    'abc', 'argparse', 'array', 'ast', 'asyncio', 'base64',
    'binascii', 'bisect', 'builtins', 'bz2', 'calendar', 'codecs',
    'collections', 'concurrent', 'configparser', 'contextlib', 'copy',
    'csv', 'ctypes', 'datetime', 'decimal', 'difflib', 'dis', 'doctest',
    'email', 'encodings', 'enum', 'errno', 'filecmp', 'fileinput',
    'fnmatch', 'functools', 'gc', 'getopt', 'getpass', 'gettext',
    'glob', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
    'inspect', 'io', 'itertools', 'json', 'keyword', 'linecache',
    'locale', 'logging', 'math', 'mimetypes', 'mmap', 'multiprocessing',
    'netrc', 'numbers', 'operator', 'optparse', 'os', 'pathlib',
    'pickle', 'platform', 'pprint', 'profile', 'pstats', 'pty',
    'queue', 'random', 're', 'reprlib', 'runpy', 'sched', 'secrets',
    'select', 'shelve', 'shlex', 'shutil', 'signal', 'socket',
    'socketserver', 'sqlite3', 'ssl', 'stat', 'statistics', 'string',
    'struct', 'subprocess', 'sys', 'sysconfig', 'tarfile', 'tempfile',
    'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token',
    'traceback', 'types', 'typing', 'unicodedata', 'unittest', 'urllib',
    'uuid', 'venv', 'warnings', 'weakref', 'webbrowser', 'xml', 'xmlrpc',
    'zipfile', 'zipimport', 'zlib'
])

def get_imported_libs(project_root='.'):
    """
    扫描项目所有 .py 文件，提取顶层导入的第三方库名。
    """
    imported = set()
    py_files = Path(project_root).rglob('*.py')
    for py_file in py_files:
        if py_file.name.startswith('install_deps'):
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # 匹配 import xxx 或 from xxx import yyy
            matches = re.findall(r'^(?:from|import)\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
            for m in matches:
                if m not in STDLIB and not m.startswith('_'):
                    imported.add(m)
        except Exception:
            continue
    return imported

def check_and_install(packages, mirror=None):
    """
    检查每个包是否已安装，若缺失则通过 pip 安装。
    返回 (已安装列表, 缺失并已安装列表, 失败列表)
    """
    installed = []
    missing = []
    failed = []

    print("=" * 60)
    print("开始检查依赖库...")
    print("=" * 60)

    # 映射包名到实际 pip 包名（如 scikit-learn 对应 sklearn）
    pip_name_map = {
        'sklearn': 'scikit-learn',
        'cvxpy': 'cvxpy',
        'statsmodels': 'statsmodels',
        'lightgbm': 'lightgbm',
        'pytz': 'pytz',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'tabulate': 'tabulate',
    }

    for pkg in packages:
        if pkg in STDLIB:
            continue
        pip_name = pip_name_map.get(pkg, pkg)
        try:
            importlib.import_module(pkg)
            print(f"✓ {pkg} 已安装")
            installed.append(pkg)
        except ImportError:
            print(f"✗ {pkg} 未安装，正在安装...")
            missing.append(pkg)
            cmd = [sys.executable, '-m', 'pip', 'install']
            if mirror:
                cmd.extend(['-i', mirror])
            cmd.append(pip_name)
            try:
                subprocess.check_call(cmd, stderr=subprocess.STDOUT)
                print(f"  ✅ {pkg} 安装成功")
                installed.append(pkg)
            except subprocess.CalledProcessError:
                print(f"  ❌ {pkg} 安装失败，请手动安装")
                failed.append(pkg)

    return installed, missing, failed

def main():
    parser = argparse.ArgumentParser(description='Quant-Ultra 依赖自动安装')
    parser.add_argument('--mirror', type=str, default=None,
                        help='指定 pip 镜像源，如 https://pypi.tuna.tsinghua.edu.cn/simple')
    parser.add_argument('--scan', action='store_true',
                        help='扫描项目代码自动识别依赖（推荐）')
    args = parser.parse_args()

    if args.scan:
        print("正在扫描项目源代码中的导入...")
        libs = get_imported_libs()
        third_party = [lib for lib in libs if not lib.startswith('step') and not lib.startswith('test')]
        packages = list(set(REQUIRED_LIBRARIES + third_party))
    else:
        packages = REQUIRED_LIBRARIES

    print(f"将检查以下库: {', '.join(packages)}")

    installed, missing, failed = check_and_install(packages, args.mirror)

    print("\n" + "=" * 60)
    print("安装结果汇总")
    print("=" * 60)
    print(f"已安装/成功: {len(installed)} 个")
    if missing:
        print(f"原缺失但已安装: {len(missing)} 个")
    if failed:
        print(f"安装失败: {len(failed)} 个")
        print("请根据错误信息手动安装这些库。")
        sys.exit(1)
    else:
        print("所有依赖库均已就绪，可以启动项目。")
        sys.exit(0)

if __name__ == '__main__':
    main()