from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

image_path = 'C:\\Users\\ijan\\Desktop\\Udacity\\CarND-Object-Detection-Lab\\assets\\training_img-1.jpeg'
im = Image.open(image_path)

image_np = load_image_into_numpy_array(im)

plt.imshow(image_np)
plt.show()