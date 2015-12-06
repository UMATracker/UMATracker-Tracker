import os


datas = [('./data', 'data'),]

binaries = [(os.path.join(os.getcwd(), 'dll', 'ffms2.dll'), 'dll'),
        (os.path.join(os.getcwd(), 'dll', 'msvcp120.dll'), 'dll'),
        (os.path.join(os.getcwd(), 'dll', 'msvcr120.dll'), 'dll'),
        (os.path.join(os.getcwd(), 'dll', 'opencv_ffmpeg300_64.dll'), 'dll')]

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
        ("llvmlite/binding/llvmlite.dll", "./dll/llvmlite.dll", 'BINARY'),
        ("llvmlite/binding/msvcp140.dll", "./dll/msvcp140.dll", 'BINARY'),
        ("llvmlite/binding/vcruntime140.dll", "./dll/vcruntime140.dll", 'BINARY'),
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
