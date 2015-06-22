import os.path

currentDirPath = os.path.abspath(os.path.dirname(__file__) )

pythonLibDirPath = os.path.join(currentDirPath, 'lib', 'python')
sampleDataPath = os.path.join(currentDirPath,"data")
userDir        = os.path.expanduser('~')

blocklyURL = "file:///{0}".format(os.path.join(currentDirPath,"lib","editor","index.html").replace("\\","/"))
