import tensorflow  as tf
import matplotlib.pyplot as plt


image=tf.read_file("C:\\Users\\681\\code\\my_git_code\\homework\\image_03151.jpg",'r')

image_tensor=tf.image.decode_image(image)

shape=tf.shape(image_tensor)

session=tf.Session()

print(session.run(shape))
image_ndarray=image_tensor.eval(session=session)
print(image_ndarray)

plt.imshow(image_ndarray)
plt.show()



