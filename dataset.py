import numpy as np
from matplotlib import pyplot
from tensorflow import keras

class Mnist:
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = \
            keras.datasets.mnist.load_data()

    def plot_imgs(self, rows: int = 5, columns: int = 5):
        """
        Draws a rows * columns element plot from the Mnist dataset

        Parameters
        ----------
        rows: int
            Number of rows for the plot
        columns: int
            Number of columns for the plot
        """
        for i in range(rows * columns):
            # Define subplot
            pyplot.subplot(rows, columns, 1 + i)
            # Turn off axis because it cluters the image
            pyplot.axis('off')
            # Show the image
            pyplot.imshow(self.train_x[i], cmap='gray_r')

        pyplot.show()


    def preprocess_train_data(self):
        # Expand to 3D, add a new dimension for channels
        train_x_3d = np.expand_dims(self.train_x, axis = -1)
        # Convert from unsigned ints to floats
        train_x_3d = train_x_3d.astype('float32')
        # Scale from [0, 255] range to [0, 1] range
        train_x_3d = train_x_3d / 255.0

        return train_x_3d



if __name__ == "__main__":
    mnist = Mnist()
    mnist.plot_imgs(rows=10, columns=8)

