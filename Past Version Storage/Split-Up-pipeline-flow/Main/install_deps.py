#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quant-Ultra + Conformal-BL 项目依赖自动检查与安装脚本（增强版）
用法: python install_deps.py [--mirror https://pypi.tuna.tsinghua.edu.cn/simple] [--scan] [--no-upgrade]
特性: 每次运行自动更新 pip 和所有已安装库，额外安装 akshare、baostock
"""

import subprocess
import sys
import importlib
import re
import os
import argparse
from pathlib import Path

import pandas_market_calendars

# 项目所需的核心第三方库（根据代码实际导入情况整理 + 新增）
REQUIRED_LIBRARIES = [
    "numpy",
    "pandas",
    "pytz",
    "lightgbm",
    "scikit-learn",
    "statsmodels",
    "scipy",
    "cvxpy",
    "matplotlib",
    "tabulate",
    "pandas_market_calendars",
    "pyyaml",
    "requests",
    "pyarrow",          # 数据处理库   
    'psutil',           # 系统监控库
    "akshare",          # 金融数据接口
    "baostock",         # 金融数据接口
    "tushare",          # 金融数据接口
    'efinance',         # 金融数据接口
    'yfinance',         # 金融数据接口
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
    imported = set()
    py_files = Path(project_root).rglob('*.py')
    for py_file in py_files:
        if py_file.name.startswith('install_deps'):
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            matches = re.findall(r'^(?:from|import)\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
            for m in matches:
                if m not in STDLIB and not m.startswith('_'):
                    imported.add(m)
        except Exception:
            continue
    return imported

def upgrade_pip_and_all_packages(mirror=None):
    """升级 pip 以及所有已安装的第三方库"""
    print("=" * 60)
    print("开始升级 pip 和所有已安装的第三方库...")
    print("=" * 60)

    # 1. 升级 pip 自身
    pip_cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip']
    if mirror:
        pip_cmd.extend(['-i', mirror])
    try:
        subprocess.check_call(pip_cmd)
        print("✓ pip 升级成功")
    except subprocess.CalledProcessError as e:
        print(f"✗ pip 升级失败: {e}")

    # 2. 获取所有已安装的第三方库（排除标准库、setuptools、wheel、pip 本身等）
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
            capture_output=True, text=True, check=True
        )
        installed_packages = []
        for line in result.stdout.splitlines():
            pkg_name = line.split('==')[0].strip()
            # 跳过一些基础包，避免冲突
            if pkg_name.lower() in ('pip', 'setuptools', 'wheel', 'distribute'):
                continue
            installed_packages.append(pkg_name)
        print(f"发现 {len(installed_packages)} 个第三方库，开始逐个升级...")
    except subprocess.CalledProcessError as e:
        print(f"✗ 获取已安装列表失败: {e}")
        return

    # 3. 逐个升级（若指定镜像则使用）
    success_count = 0
    fail_count = 0
    for pkg in installed_packages:
        print(f"  升级 {pkg} ...", end=f" 已完成 {success_count + fail_count} ")
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade']
        if mirror:
            cmd.extend(['-i', mirror])
        cmd.append(pkg)
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✓")
            success_count += 1
        except subprocess.CalledProcessError:
            print("✗")
            fail_count += 1

    print(f"升级完成: 成功 {success_count} 个，失败 {fail_count} 个")
    return success_count, fail_count

def check_and_install(packages, mirror=None):
    installed = []
    missing = []
    failed = []

    print("=" * 60)
    print("开始检查依赖库是否完整安装...")
    print("=" * 60)

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
        'pandas_market_calendars': 'pandas_market_calendars',
        'yaml': 'pyyaml',
        'requests': 'requests',
        'pyarrow': 'pyarrow',
        'psutil': 'psutil',
        'akshare': 'akshare',      # 新增
        'baostock': 'baostock',    # 新增
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
    parser = argparse.ArgumentParser(description='Quant-Ultra 依赖自动安装（增强版）')
    parser.add_argument('--mirror', type=str, default=None,
                        help='指定 pip 镜像源，如 https://pypi.tuna.tsinghua.edu.cn/simple')
    parser.add_argument('--scan', action='store_true',
                        help='扫描项目代码自动识别依赖（推荐）')
    parser.add_argument('--no-upgrade', action='store_true',
                        help='跳过全局 pip 和所有库的升级（默认会进行升级）')
    args = parser.parse_args()

    # 1. 升级 pip 和所有已安装库（除非用户明确跳过）
    if not args.no_upgrade:
        upgrade_pip_and_all_packages(args.mirror)
    else:
        print("已跳过全局升级（--no-upgrade）")

    # 2. 确定需要检查的包列表
    if args.scan:
        print("正在扫描项目源代码中的导入...")
        libs = get_imported_libs()
        third_party = [lib for lib in libs if not lib.startswith('step') and not lib.startswith('test')]
        packages = list(set(REQUIRED_LIBRARIES + third_party))
    else:
        packages = REQUIRED_LIBRARIES

    print(f"将检查以下库: {', '.join(packages)}")

    # 3. 检查并安装缺失的库
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