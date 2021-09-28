from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

import tensorflow as tf


seed = 5
tf.random.set_seed(seed)
np.random.seed(seed)
#gpu 메모리 할당
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


caltech_dir = './test_data'


image_w = 128
image_h = 128



X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    filenames.append(f)
    X.append(data)


X = np.array(X)
X = X.astype(float) / 255
model = load_model('./model/kongju_others_classify.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0
for i in prediction:
    if i >= 0.3: print("해당 " + filenames[cnt].split("\\")[0] + filenames[cnt].split("\\")[1] + "  이미지는 공주대가 아닌것으로 추정됩니다.")
    else : print("해당 " + filenames[cnt].split("\\")[0] + filenames[cnt].split("\\")[1] + "  이미지는 공주대로 추정됩니다.")
    cnt += 1
