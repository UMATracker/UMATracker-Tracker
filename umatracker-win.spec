import os


datas = [('./data', 'data'),]

binaries = []
for dir_path, dir_names, file_names in os.walk("dll"):
    for file_name in file_names:
        binaries.append((os.path.join('.\\', dir_path, file_name), 'dll'))

a = Analysis(['./main.py'],
        pathex=['./'],
        binaries=binaries,
        datas=datas,
        hiddenimports=[],
        hookspath=['./hooks',],
        runtime_hooks=None,
        excludes=None,
        win_no_prefer_redirects=None,
        win_private_assemblies=None,
        cipher=None)

a.binaries += [
        ("llvmlite/binding/llvmlite.dll", "./dll/LLVMLite/llvmlite.dll", 'BINARY'),
        ("llvmlite/binding/msvcp140.dll", "./dll/LLVMLite/msvcp140.dll", 'BINARY'),
        ("llvmlite/binding/vcruntime140.dll", "./dll/LLVMLite/vcruntime140.dll", 'BINARY'),
        ]

pyz = PYZ(a.pure, cipher=None)

exe = EXE(pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        a.binaries,
        name='UMATracker-DetectCenter',
        debug=False,
        strip=None,
        upx=False,
        console=False, icon='./icon/icon.ico')
