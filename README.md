# A GAN that generates images from the MNIST dataset
The model has been trained for 91 epochs.

## Train
You can train the model again using the following command.

<code>python3 gan.py -t</code>

## Generate
You can generate new MNIST images using one of the saved models. The images
will be plotted in the `generated_mnist.png` file.

<code>python3 gan.py -g -m "save_models/generated_model091.h5"</code>

## Results
You can see the the saved models after each 10 epochs in the `saved_models`
directory.
You can also see some samples of the generated imaged in the `png` directory

