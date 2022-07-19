from __future__ import print_function, division
import os, gzip, cv2
import six.moves.cPickle as pickle
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras import losses
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

class SGAN:
    def __init__(self):
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 12
        self.latent_dim = 100

        gen_optimizer = Adam(lr=0.0002, beta_1=0.5)
        disc_optimizer = Adam(lr=0.0001, beta_1=0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=disc_optimizer,
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(32 * 28 * 28, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((28, 28, 32)))
        model.add(UpSampling2D())

        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.5))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Convolution2D(96, (7,7), strides=3, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=.001))
        model.add(Convolution2D(96, (1, 1)))
        model.add(LeakyReLU(alpha=.001))
        
        model.add(Convolution2D(192, (3, 3)))
        model.add(LeakyReLU(alpha=.001))
        
        model.add(Convolution2D(192, (1, 1)))
        model.add(LeakyReLU(alpha=.001))
    
        model.add(MaxPooling2D(pool_size=(3,3)))
        #model.add(Dropout(0.25))
        model.add(Convolution2D(384, (3, 3)))
        model.add(LeakyReLU(alpha=.001))
        
        model.add(Convolution2D(384, (1, 1)))
        model.add(LeakyReLU(alpha=.001))
        
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Convolution2D(950, (1,1)))
        model.add(LeakyReLU(alpha=.001))
        
        model.add(Dropout(0.5))
        model.add(Convolution2D(12, (1,1)))
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        

        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=["categorical_accuracy"])
        #model.load_weights("Best-117.hdf5")
        
        model.pop()
        model.pop()
        model.pop()
        model.pop()

        img = Input(shape=self.img_shape)

        features = model(img)
        valid = Flatten()(features)
        valid = Dense(1, activation="sigmoid")(valid)
        
        label = Conv2D(self.num_classes+1, kernel_size=1, activation='relu')(features)
        label = GlobalAveragePooling2D()(label)
        label = Activation('softmax')(label)
        model.summary()
        return Model(img, [valid, label])

    def train(self, seed, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        print ('Loading the images from the dataset...')
        dataset = 'beans_all224.gz'
        with gzip.open(dataset, 'rb') as f:
            try:
                X_train, y_train = pickle.load(f, encoding = 'latin1') 
                _, _ = pickle.load(f, encoding='latin1')
            except:
                X_train, y_train = pickle.load(f)
                _, _ = pickle.load(f)

        # Rescale -1 to 1

        print(X_train.shape)
        X_train = 2*(X_train - np.min(X_train))/np.ptp(X_train)-1
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # One-hot encoding of labels
            labels = to_categorical(y_train[idx], num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, seed)
                self.save_model()

    def sample_images(self, epoch, seed):
        r, c = 4, 4
        
        gen_imgs = self.generator.predict(seed)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(cv2.cvtColor(gen_imgs[cnt, :,:,:], cv2.COLOR_BGR2RGB))
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/beans_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "beans_sgan_generator")
        save(self.discriminator, "beans_sgan_discriminator")
        save(self.combined, "beans_sgan_adversarial")


if __name__ == '__main__':
    sgan = SGAN()
    seed = np.random.normal(0, 1, (4 * 4, 100))
    sgan.train(seed, epochs=15500, batch_size=32, sample_interval=100)

