import cv2

image_path = "../image.png"
image = cv2.imread(image_path)

if image is None:
    print("Could not open or find the image")
    exit()

cv2.imshow("Display window", image)

k = cv2.waitKey(0)
cv2.destroyAllWindows()