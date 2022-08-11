import cv2


image = cv2.imread("image001.jpg")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray_image.jpg", image_gray)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

faces = face_cascade.detectMultiScale(image_gray)

for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

cv2.imwrite("face_edited.jpg", image)