import os
import random
import numpy as np
from keras import layers as L
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import tqdm
import cv2


epochs = 1000
batch_size = 64
dpi = 1
gpi = 1
steps_per_epoch = 400


def create_generator():
    img = L.Input(shape=(100,))
    x = L.Dense(units=4*4*1024)(img)
    x = L.LeakyReLU()(x)
    x = L.Reshape(target_shape=(4, 4, 1024))(x)
    x = L.Conv2DTranspose(filters=512, kernel_size=(5, 5),
                          padding='same', strides=(1, 1))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=256, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=128, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=64, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=3, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.Activation('tanh')(x)
    gen = Model(inputs=img, outputs=x)
    return gen


def create_discriminator():
    inp = L.Input(shape=(64, 64, 3))
    x = L.Conv2D(filters=64, kernel_size=(5, 5), padding='same', strides=(2,2))(inp)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=(2,2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=256, kernel_size=(5, 5), padding='same', strides=(2,2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=512, kernel_size=(5, 5), padding='same', strides=(2,2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=1024, kernel_size=(5, 5), padding='same', strides=(2,2))(x)
    x = L.LeakyReLU()(x)
    x = L.Flatten()(x)
    x = L.Dense(1)(x)
    x = L.Activation('sigmoid')(x)
    dis = Model(inputs=inp, outputs=x)
    dis.compile(loss=binary_crossentropy,
                optimizer=Adam(0.0002, beta_1=0.5, epsilon=1e-8))
    return dis


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = L.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss=binary_crossentropy,
                optimizer=Adam(0.0002, beta_1=0.5, epsilon=1e-8))
    return gan


generator = create_generator() #load_model('/kaggle/input/checkpoint-100-facegan/facegeneratorep100.hdf5')
discriminator = create_discriminator() #load_model('/kaggle/input/checkpoint-100-facegan/facediscriminatorep100.hdf5')
gan = create_gan(discriminator, generator)

print(gan.summary())

outputs_list = os.listdir('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba')
for e in range(1, epochs+1):
    discriminator_loss = 0
    gan_loss = 0
    print("Epoch %d" % e)
    for step in tqdm.tqdm_notebook(range(steps_per_epoch)):
        for _ in range(dpi):
            noise = np.random.normal(0, 1, [batch_size, 100])
            generated_images = generator.predict(noise)
            image_batch = []
            ls = random.sample(outputs_list,k=batch_size)
            for file in ls:
                img_path1 = str('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + str(file))
                pic1 = cv2.imread(img_path1)
                pic1 = cv2.resize(pic1, (64, 64))
                image_batch.append(pic1)
            image_batch = np.array(image_batch)
            image_batch = image_batch.reshape((-1, 64, 64, 3))
            image_batch = (image_batch.astype(float)-127.5)/127.5
            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[batch_size:] = y_dis[batch_size:] + \
                np.random.random_sample(batch_size)*0.2
            y_dis[:batch_size] = 1
            y_dis[:batch_size] = y_dis[:batch_size] - \
                np.random.random_sample(batch_size)*0.2 #label smoothing
            discriminator.trainable = True
            discriminator_loss += discriminator.train_on_batch(X, y_dis)
        for _ in range(gpi):
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan_loss += gan.train_on_batch(noise, y_gen)
    pr = generator.predict(np.random.normal(0, 1, [1, 100]))
    pr = pr.reshape(64, 64, 3)
    pr = ((pr*127.5)+127.5).astype(int)
    cv2.imwrite('/kaggle/working/epoch_gen_imgs/nbprface'+str(e)+'.jpg', pr)
    print("Discriminator loss="+str(discriminator_loss/(dpi*steps_per_epoch)))
    print("GAN loss="+str(gan_loss/(gpi*steps_per_epoch)))
    if e%5 == 0:
        generator.save('/kaggle/working/checkpoint_models/nbfacegeneratorep'+str(e)+'.hdf5')
        discriminator.save('/kaggle/working/checkpoint_models/nbfacediscriminatorep'+str(e)+'.hdf5')
generator.save('/kaggle/working/nbfacegenerator.hdf5')
discriminator.save('/kaggle/working/nbfacediscriminator.hdf5')
pr = generator.predict(np.random.normal(0, 1, [1, 100]), batch_size=1)
pr = pr.reshape(64, 64, 3)
pr = ((pr*127.5)+127.5).astype(int)
print(pr)
cv2.imwrite('/kaggle/working/face.jpg', pr)
