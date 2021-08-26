import os
import streamlit as st
st.title('Artformer')
st.header('Using Deep Learning to Transform an Image in the Style of Another Image')
st.subheader('This tool can be used to turn your images into beautiful paintings of the style of Van Gogh, Picasso, or any art piece you love!')
st.text('Paste in links for an image to transform and an image for the style. Links must end in .jpg or .png')
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display

import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  st.image(PIL.Image.fromarray(tensor))
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

with st.form(key='my_form'):
    text_input = st.text_input(label='Link for image you want to transform', key = "last_name3")
    text_input1 = st.text_input(label='Link for image with the style you want', key = "last_name2")
    submit_button = st.form_submit_button('Submit')
def img1(text_input):
    content_path1 = tf.keras.utils.get_file(text_input.split("/")[-1], str(text_input))
    content_image1 = load_img(content_path1)
    return content_image1

if submit_button:
   stylized_image = hub_model(tf.constant(img1(text_input)), tf.constant(img1(text_input1)))[0]
   tensor_to_image(stylized_image)
   print('dog')

