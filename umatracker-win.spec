import os
from distutils.sysconfig import get_python_lib

datas = [('./data', 'data'),
        ('./lib/python/tracking_system', 'lib/python/tracking_system'),]

a = Analysis(['./main.py'],
        pathex=['./'],
        binaries=None,
        datas=datas,
        hiddenimports=[],
        hookspath=['./hooks',],
        runtime_hooks=None,
        excludes=None,
        win_no_prefer_redirects=None,
        win_private_assemblies=None,
        cipher=None)


# Additional DLLs
tmp = []
for dir_path, dir_names, file_names in os.walk("dll"):
    for file_name in file_names:
        tmp.append(
                (
                    file_name,
                    os.path.join(os.getcwd(), dir_path, file_name),
                    'BINARY'
                    )
                )

# For LLVMLite
llvmlite_dll_path = os.path.join(get_python_lib(), 'llvmlite')
for dir_path, dir_names, file_names in os.walk(llvmlite_dll_path):
    for file_name in file_names:
        if os.path.splitext(file_name)[1]=='.dll':
            tmp.append(
                    (
                        os.path.join('llvmlite', 'binding', file_name),
                        os.path.join(dir_path, file_name),
                        'BINARY'
                        )
                    )

# For Numpy MKL
blacklist = ['mkl_rt.dll', 'tbb.dll', 'libmmd.dll', 'libifcoremd.dll']
a.binaries = list(filter(lambda t:t[0] not in blacklist, a.binaries))
numpy_dll_path = os.path.join(get_python_lib(), 'numpy', 'core')
for dir_path, dir_names, file_names in os.walk(numpy_dll_path):
    for file_name in file_names:
        if os.path.splitext(file_name)[1]=='.dll':
            tmp.append(
                    (
                        file_name,
                        os.path.join(dir_path, file_name),
                        'BINARY'
                        )
                    )

a.binaries += tmp

pyz = PYZ(a.pure, cipher=None)

exe = EXE(pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        a.binaries,
        name='UMATracker-DetectCenter',
        debug=True,
        strip=None,
        upx=False,
        console=True, icon='./icon/icon.ico')
