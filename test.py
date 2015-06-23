def filterFunc(img):
    return cv2.threshold(cv2.cvtColor(filters.Filter.colorFilter(img,[255,0,0],80),cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)[1]
