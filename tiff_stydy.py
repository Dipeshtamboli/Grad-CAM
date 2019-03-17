import cv2	

add = "/home/dipesh/grad_cam/data/test/airplane/airplane80.tif"
# image = Image.open(add)
image = cv2.imread(add, cv2.IMREAD_UNCHANGED)
# image.imshow()
print image.shape
cv2.imshow("img",image)