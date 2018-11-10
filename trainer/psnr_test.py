import tensorflow as tf

# Read images from file.
im1 = tf.decode_png('path/to/im1.png')
im2 = tf.decode_png('path/to/im2.png')
# Compute PSNR over tf.uint8 Tensors.
psnr1 = tf.image.psnr(im1, im2, max_val=255)

print(psnr1)