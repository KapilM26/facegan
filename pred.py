import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from keras.models import load_model
import cv2
import numpy as np

noise = []
noise = np.random.normal(0, 1, [100, 100])
noise = np.array(noise)
print(noise.shape)
model = load_model('facegeneratorep100.hdf5')
pr = model.predict(noise)
pr = ((pr*127.5)+127.5).astype(int)
for i, img in enumerate(pr):
    cv2.imwrite('predictions/'+str(i)+'.jpg', img)
