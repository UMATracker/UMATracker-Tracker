import os
import distutils.sysconfig

datas = [('./data', 'data'),
        ('./lib/python/tracking_system', 'lib/python/tracking_system'),]

binaries = [(r'/usr/local/Cellar/ffms2/2.21/lib/libffms2.dylib', 'lib'), ]

site_package_dir = distutils.sysconfig.get_python_lib()
a = Analysis(['./main.py'],
            pathex=['./'],
            binaries=binaries,
            datas=datas,
            hiddenimports=['sklearn', 'numpy', 'numba'],
            hookspath=['./hooks',],
            runtime_hooks=None,
            excludes=None,
            win_no_prefer_redirects=None,
            win_private_assemblies=None,
            cipher=None)
a.binaries += [("llvmlite/binding/libllvmlite.dylib", os.path.join(site_package_dir, "llvmlite/binding/libllvmlite.dylib"), 'BINARY')]

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
        upx=True,
        console=False, icon='./icon/icon.icns')

coll = COLLECT(exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=None,
        upx=True,
        name=os.path.join('dist', 'UMATracker'))

app = BUNDLE(coll,
        name=os.path.join('dist', 'UMATracker-DetectCenter.app'),
        appname="UMATracker-DetectCenter",
        version = '0.1', icon='./icon/icon.icns'
        )
