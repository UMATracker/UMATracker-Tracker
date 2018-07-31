import os
import distutils.sysconfig
import glob

datas = [('./data', 'data'),
        ('./lib/python/tracking_system', 'lib/python/tracking_system'),]

dlls = glob.glob('/usr/local/Cellar/ffms2/*/lib/libffms2.dylib')

binaries = [
    (x, 'lib')
    for x in dlls
]

site_package_dir = distutils.sysconfig.get_python_lib()
a = Analysis(['./main.py'],
            pathex=['./'],
            binaries=binaries,
            datas=datas,
            hiddenimports=['sklearn', 'numpy', 'shapely', 'networkx', 'fractions'],
            hookspath=['./hooks',],
            runtime_hooks=None,
            excludes=None,
            win_no_prefer_redirects=None,
            win_private_assemblies=None,
            cipher=None)

pyz = PYZ(a.pure, cipher=None)

exe = EXE(pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        a.binaries,
        name='UMATracker-Tracking',
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
        name=os.path.join('dist', 'UMATracker-Tracking.app'),
        appname="UMATracker-Tracking",
        version = '0.1', icon='./icon/icon.icns'
        )
