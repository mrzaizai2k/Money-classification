import cv2
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model


def cv2_to_pil(img):  # Since you want to be able to use Pillow (PIL)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# Load the model
model = load_model('keras_model.h5')
label = {0: '50k', 1: '100k'}
size = (224, 224)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    # Image preprocessing
    ret, img = cap.read()
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # cv2.imshow('img',img)
    image = cv2_to_pil(img)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    p = np.argmax(prediction)

    cv2.putText(resized_img, 'result: {}'.format(label[p]), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    cv2.imshow('prediction', resized_img)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
