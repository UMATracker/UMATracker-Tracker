import os
from distutils.sysconfig import get_python_lib

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


# Additional DLLs
tmp = []

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
                        os.path.join('numpy', 'core', file_name),
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
        debug=False,
        strip=None,
        upx=False,
        console=False, icon='./icon/icon.ico')
