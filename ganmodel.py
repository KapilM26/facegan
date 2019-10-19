import os
import random
import cv2
import numpy as np
from keras import layers as L
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, SGD
from tqdm import tqdm

epochs = 400
batch_size = 32
dpi = 2
gpi = 1
steps_per_epoch = 200


def create_generator():
    img = L.Input(shape=(100,))
    x = L.Dense(units=4*4*512)(img)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU()(x)
    x = L.Reshape(target_shape=(4, 4, 512))(x)
    x = L.Conv2DTranspose(filters=512, kernel_size=(5, 5),
                          padding='same', strides=(1, 1))(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=256, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=128, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=64, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=3, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.Activation('tanh')(x)
    gen = Model(inputs=img, outputs=x)
    gen.compile(loss=binary_crossentropy,
                optimizer=Adam(0.0001, epsilon=1e-8))
    return gen


def create_discriminator():
    inp = L.Input(shape=(64, 64, 3))
    x = L.Conv2D(filters=32, kernel_size=(3, 3), padding='valid')(inp)
    x = L.LeakyReLU()(x)
    x = L.Dropout(0.2)(x)
    x = L.Conv2D(filters=64, kernel_size=(3, 3), padding='valid')(x)
    x = L.LeakyReLU()(x)
    x = L.Dropout(0.2)(x)
    x = L.Conv2D(filters=128, kernel_size=(3, 3), padding='valid')(x)
    x = L.LeakyReLU()(x)
    x = L.Dropout(0.2)(x)
    x = L.Conv2D(filters=256, kernel_size=(3, 3), padding='valid')(x)
    x = L.LeakyReLU()(x)
    x = L.Dropout(0.2)(x)
    x = L.Flatten()(x)
    x = L.Dense(1, activation='sigmoid')(x)
    dis = Model(inputs=inp, outputs=x)
    dis.compile(loss=binary_crossentropy,
                optimizer=SGD(0.0001))
    return dis


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = L.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss=binary_crossentropy,
                optimizer=Adam(0.0001, epsilon=1e-8))
    return gan


# load_model('checkpoint_models/facegeneratorep29.hdf5')
generator = create_generator()
# load_model('checkpoint_models/facediscriminatorep29.hdf5')
discriminator = create_discriminator()
gan = create_gan(discriminator, generator)

print(gan.summary())

outputs_list = os.listdir('../celeba_resized')
for e in range(1, epochs+1):
    discriminator_loss = 0
    gan_loss = 0
    print("Epoch %d" % e)
    for _ in tqdm(range(steps_per_epoch)):
        for _ in range(dpi):
            noise = np.random.normal(0, 1, [batch_size, 100])
            generated_images = generator.predict(noise)
            image_batch = []
            ls = random.sample(outputs_list, k=batch_size)
            for file in ls:
                img_path1 = str('../celeba_resized/' + str(file))
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
                np.random.random_sample(batch_size)*0.2
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
    cv2.imwrite('epoch_gen_imgs/prface'+str(e)+'.jpg', pr)
    print("Discriminator loss="+str(discriminator_loss/(dpi*steps_per_epoch)))
    print("GAN loss="+str(gan_loss/(gpi*steps_per_epoch)))
    generator.save('checkpoint_models/facegeneratorep'+str(e)+'.hdf5')
    discriminator.save('checkpoint_models/facediscriminatorep'+str(e)+'.hdf5')
generator.save('facegenerator.hdf5')
discriminator.save('facediscriminator.hdf5')
pr = generator.predict(np.random.normal(0, 1, [1, 100]), batch_size=1)
pr = pr.reshape(64, 64, 3)
pr = ((pr*127.5)+127.5).astype(int)
print(pr)
cv2.imwrite('face.jpg', pr)
