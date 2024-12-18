import tensorflow as tf
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

class Logger:
    """Logging in tensorboard using TensorFlow 2.x summary API."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Parameters
        ----------
        tag : str
            Name of the scalar
        value : float
            Scalar value to log
        step : int
            Training iteration
        """
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def log_images(self, tag, images, step, post_tag=None):
        """Logs a list of images."""

        with self.writer.as_default():
            for nr, img in enumerate(images):
                s = BytesIO()
                plt.imsave(s, img, format='jpg')
                img_tensor = tf.image.decode_jpeg(s.getvalue())
                img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension

                tf.summary.image(name=f"{tag}/{nr}/{post_tag}", data=img_tensor, step=step)
            self.writer.flush()

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        
        with self.writer.as_default():
            values = np.array(values)
            counts, bin_edges = np.histogram(values, bins=bins)
            
            # Log the histogram using tf.summary.histogram
            tf.summary.histogram(name=tag, data=values, step=step)
            self.writer.flush()
