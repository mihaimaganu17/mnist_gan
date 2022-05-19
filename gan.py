import numpy as np
from matplotlib import pyplot
from tensorflow.keras import models, layers, optimizers, utils

class Discriminator:
    def __init__(self, input_shape=(28, 28, 1)):
        self.model = models.Sequential()
        model_layers = [
            layers.Conv2D(64, (3, 3), strides=(2,2), padding='same',
                input_shape=input_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ]

        for layer in model_layers:
            self.model.add(layer)

        # Add optimizations
        opt = optimizers.Adam(lr=0.0002, beta_1=0.5)

        self.model.compile(loss='binary_crossentropy', optimizer=opt,
                metrics=['accuracy'])


    def plot_model(self):
        self.model.summary()
        utils.plot_model(
                self.model,
                to_file="discriminator_plot.png",
                show_shapes=True,
                show_layer_names=True
        )

    def model():
        return self.model

    def train(self, dataset, iterations=100, batch_size=256):
        half_batch_size = int(batch_size / 2)
        # For each epoch/iteration
        for i in range(iterations):
            # Get randomly selected `real` samples
            x_real, y_real = generate_real_samples(dataset, half_batch_size)
            # Update discriminator on real samples
            _, real_acc = self.model.train_on_batch(x_real, y_real)
            # Generate `fake` samples
            x_fake, y_fake = generate_fake_samples(half_batch_size)
            # Update discriminator on fake samples
            _, fake_acc = self.model.train_on_batch(x_fake, y_fake)
            # Sumarrize performance
            print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100,
                fake_acc*100))



def generate_real_samples(dataset, n_samples):
    """
    Generates real samples to be fed in the discriminator

    Parameters
    ----------
    dataset: Train data where each item is with a 3D shape of (28, 28, 1)
        from MNIST handwritten digit dataset
    n_samples: Number of samples to generate

    Return
    ------
    A tuple containing a fraction of the `dataset` described above and an
    array of `real` labels with size `n_samples`
    """
    # Chose random instances
    random_indexes = np.random.randint(0, dataset.shape[0], n_samples)
    # Retrieve selected images
    random_images = dataset[random_indexes]
    # Generate `real` class labels as ones
    labels = np.ones((n_samples, 1))
    # Return tuple
    return random_images, labels


def generate_fake_samples(model, latent_len, n_samples):
    """
    Generate fake MNIST image samples to be fed into the discriminator

    Parameters
    ----------
    n_samples: int
        Number of samples to be generated
    Return
    ------
    A tuple containing a `n_samples` fake samples and a vector of zeroes with
    size `n_samples` representing `fake` labels
    """
    # Draw latent points
    latent_points = generate_latent_points(latent_len, n_samples)
    # Feed the input to the generator, which by predicting, is basically
    # generating a new fake image sample
    fake_samples = model.predict(latent_points)
    # Generate 'fake' class labels
    labels = np.zeros((n_samples, 1))

    return fake_samples, labels


class Generator:
    def __init__(self, latent_space_size):
        """
        Creates a generator model
        """
        self.model = models.Sequential()
        # Create a dense layer that simulates a feature map with 128 filters,
        # each of size 7 by 7
        # It receives a 100 element vector of Gaussian random numbers.
        self.model.add(layers.Dense(128 * 7 * 7, input_dim = latent_space_size))
        # Use LeakyReLU activation with the best activation slope for GANs
        # of 0.2
        self.model.add(layers.LeakyReLU(alpha=0.2))
        # We reshape it into 128 feature maps
        self.model.add(layers.Reshape((7, 7, 128)))
        # Upsample the low_resolution image to a higher resolution version of
        # the image.
        # Use a stride of (2x2) that will quadruple the area of the input
        # feature maps.
        # A good practice is to use a kernel size that is a factor of the
        # stride(double) to avoid the checkerboard pattern
        # Here we upsample it from 7x7 to 14x14
        self.model.add(layers.Conv2DTranspose(
            128, (4,4), strides=(2,2), padding='same'
        ))
        # Add another LeakyReLU for activation
        self.model.add(layers.LeakyReLU(alpha=0.2))
        # Upsample again from 14x14 to 28x28
        self.model.add(layers.Conv2DTranspose(
            128, (4,4), strides=(2,2), padding='same'
        ))
        # Add activation
        self.model.add(layers.LeakyReLU(alpha=0.2))
        # Output layer will be a Conv2D with one filter and a kernel size of
        # 7x7. This will create a single feature map of 28x28
        # We use `sigmoid` as the activation function so we get outputs in the
        # desired range of [0,1]
        self.model.add(layers.Conv2D(
            1, (7,7), activation='sigmoid', padding='same'
        ))

    def summary(self):
        self.model.summary()
        utils.plot_model(
                self.model,
                to_file='generator_plot.png',
                show_shapes=True,
                show_layer_names=True
        )


def generate_latent_points(latent_len, n_samples):
    """
    Generates points for the latent space, drawn from the standard Gaussian

    Parameters
    ----------
    latent_len: int
        How many points one array should have
    n_samples: int
        How many arrays of point should be generated

    Return
    ------
    An array of shape (`n_samples`, `latent_len`) representing the latent space
    """
    random_points = np.random.randn(n_samples * latent_len)
    latent_space = random_points.reshape(n_samples, latent_len)
    return latent_space


def generator_demo(latent_len=100, n_samples=25):
    generator = Generator(latent_len)
    fake_samples, _ = generate_fake_samples(generator.model, latent_len, n_samples)
    for i in range(n_samples):
        # Divide the plot into subplots
        pyplot.subplot(5, 5, 1 + i)
        # Turn off axis
        pyplot.axis('off')
        # Plot a single image
        pyplot.imshow(fake_samples[i, :, :, 0], cmap='gray_r')
    #Show the figure
    pyplot.show()


class Gan:
    def __init__(self, generator, discriminator):
        # Define a new logical model that basically concatenates the generator
        # and the discriminator
        self.model = models.Sequential()
        # Make the discriminator weights not trainable so we only train the
        # generator
        discriminator.model.trainable = False
        # Add generator
        self.model.add(generator.model)
        # Add dicriminator
        self.model.add(discriminator.model)

        # Compile model
        opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=opt)

    def train(self, latent_size, n_epochs=100, batch_size=256):
        # Manually go through each epoch
        for e in range(n_epochs):
            # Prepare points in the latent space as input for the generator
            x_gan = generate_latent_points(latent_size, batch_size)
            # Create inverted labels for the fake samples
            # This make the model train the generator model
            y_gan = np.ones((batch_size, 1))
            # Update the generator via the discriminator's error
            gan_model.train_on_batch(x_gan, y_gan)


def train(generator, discriminator, gan, dataset, latent_size, epochs=100,
        batch_size = 256):
    """
    Main training function for the entire GAN model training

    Parameters
    ----------
    generator: Generator
        Generator model to be trained as part of the GAN model
    discriminator: Discriminator
        Discriminator model to be trained and used as part of the GAN model
    dataset: Array 3D
        Array containing grayscale images of the MNIST dataset
    epochs: int
        Number of epochs to train
    batch_size: int
        Size of one batch in each training epoch
    """
    batches_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch_size = int(batch_size / 2)

    # Manually enumerate epochs
    for e in range(epochs):
        # Enumerate batches over the training set
        for batch in range(batches_per_epoch):
            # Get randomly selected `real` samples
            x_real, y_real = generate_real_samples(dataset, half_batch_size)
            # Generate `fake` examples
            x_fake, y_fake = generate_fake_samples(generator.model, latent_size,
                    half_batch_size)
            # Merge the 2 half-batches above, creating a single batch set for
            # the discrimnator
            x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            # Update discriminator model weights
            d_loss, _ = discriminator.model.train_on_batch(x, y)
            # Prepare points in the latent space as input for the generator
            x_gan_latent = generate_latent_points(latent_size, batch_size)
            # Create inverted labels for the fake samples
            y_gan = np.ones((batch_size, 1))
            # Update the generator via the disciminator's error
            g_loss = gan.model.train_on_batch(x_gan_latent, y_gan)
            # Summarize loos on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' %
                    (e+1, batch+1, batches_per_epoch, d_loss, g_loss))
        if (e+10) % 10 == 0:
            summarize_performance(e, generator, discriminator, dataset,
                    latent_size)


def summarize_performance(epoch, generator, discriminator, dataset, latent_size,
        n_samples=100):
    """
    Summarize performance for both the generator and the discriminator models

    Parameters
    ----------
    epoch: int
        The training epoch we generate summary for
    generator: Generator
        Generator model used to generate samples for training
    discriminator: Discriminator
        Discriminator model used to compare between real and fake examples
    dataset: array 3D
        Array containing real MNIST grayscale image samples
    latent_size: int
        How many latent point we have in the latent space
    n_samples: int
        How many samples we should evaluate our models on
    """
    # Prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # Evaluate the discriminator on real samples
    _, acc_real = discriminator.model.evaluate(x_real, y_real, verbose=0)
    # Prepare fake samples
    x_fake, y_fake = generate_fake_samples(generator.model, latent_size,
            n_samples)
    # Evaluate discriminator on fake samples
    _, acc_fake = discriminator.model.evaluate(x_fake, y_real, verbose=0)
    # Summarize discriminator performance
    print(">Accuracy real: %.0f%%, fake: %.0f%%" %
            (acc_real*100, acc_fake*100))
    # Save some examples
    save_plot(x_fake, epoch)
    # Save the generator model in a file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    generator.model.save(filename)


def save_plot(examples, epoch, n=10):
    """
    Function that plots examples

    Parameters
    ----------
    examples: array 4D
        Array containing examples generated by our GAN model
    epoch: int
        Epoch number where the `examples were generated`
    n: int
        Number of rows and columns to plot
    """
    for i in range(n*n):
        # Divide plot into multiple subplots
        pyplot.subplot(n, n, i+1)
        # Turn off axis
        pyplot.axis('off')
        # Plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


from dataset import Mnist

if __name__ == "__main__":
    latent_space_size = 100
    discriminator = Discriminator()
    generator = Generator(latent_space_size)
    gan = Gan(generator, discriminator)
    dataset = Mnist().preprocess_train_data()
    train(generator, discriminator, gan, dataset, latent_space_size)
