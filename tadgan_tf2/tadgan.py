# Class of the time series reconstruction model (TadGAN) and related 
# ingredients. This is largely a re-implementation of the original work in
# https://github.com/signals-dev/Orion in TensorFlow 2. 
#
#

import tensorflow as tf
import numpy as np
import time

from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Conv1D, UpSampling1D, Cropping1D, \
    LSTM, Bidirectional, TimeDistributed, LeakyReLU, Dropout, Flatten, Reshape


def _wasserstein_loss(y_true, y_pred):  # y_true = 1 or -1
    return tf.reduce_mean(y_true * y_pred)


class _RandomWeightedMean(Layer):
    """Subclass of tf.keras.layers.Layer for computing a randomly weighted mean
    of two given tensors.
    """
    def call(self, inputs):
        x0, x1 = inputs
        t = tf.random.uniform(x0.shape[1:])
        return x0 * (1 - t) + x1 * t
    

class _GradientSquared(Layer):
    """Subclass of tf.keras.layers.Layer for computing the gradient squared of
    a function at a given tensor.
    """

    def __init__(self, fn):
        super(_GradientSquared, self).__init__()
        self.fn = fn
        
    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            val = tf.reduce_sum(self.fn(inputs))
        grad = tape.gradient(val, inputs)
        grad_sq = tf.reduce_sum(grad ** 2, axis=range(1, len(inputs.shape)))
        return grad_sq


class TadGAN(object):
    
    """Class of a GAN-based time series reconstruction model that was first 
    developed in https://arxiv.org/abs/2009.07769v3 and aims to reconstruct
    non-anomalous parts of time series.
    
    The model has the following components:
    - The encoder embeds any time series into a latent space.
    - The generator creates a time series from any vector in the latent space.
    - The input space critic evaluates a scalar score for any time series.
    - The latent space critic evaluates a scalar score for any vector in the
      latent space.
    
    Training alternates between two stages. During one stage, the input space 
    critic learns to differentiate real time series from generatated ones, and
    the latent space critic learns to differentiate random vectors in the 
    latent space from embeddings of real time series. During the other stage, 
    the encoder and generator learn to fool the two critics and also recover
    real time series together.
    
    After training, the model reconstructs time series by combining actions of
    the encoder and the generator.
    """
    
    def __init__(self, **params):
        self.input_dim = params.get('input_dim', 100)
        self.latent_dim = params.get('latent_dim', 20)
        lr = params.get('learning_rate', 0.0001)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.encoder_params = params.get('encoder', {})
        self.generator_params = params.get('generator', {})
        self.critic_x_params = params.get('critic_x', {})
        self.critic_z_params = params.get('critic_z', {})

    def _build(self):
        
        # architecture of encoder
        
        LSTM_units = self.encoder_params.get('LSTM_units', 100)
        
        self.encoder = Sequential([
            Reshape((self.input_dim, 1)),
            Bidirectional(LSTM(LSTM_units, return_sequences=True)),
            Flatten(),
            Dense(self.latent_dim)
        ])
        
        # architecture of generator 
        # Note: Unlike the original implementation, recurrent dropout is turned
        # off for the use of cuDNN kernel.

        int_dim, crop_dim = (self.input_dim + 1) // 2, self.input_dim % 2
        LSTM_units = self.generator_params.get('LSTM_units', 64)
        LSTM_dropout = self.generator_params.get('LSTM_dropout', 0.2)
        
        self.generator = Sequential([
            Dense(int_dim),
            Reshape((int_dim, 1)),
            Bidirectional(LSTM(LSTM_units, return_sequences=True, dropout=LSTM_dropout)),
            UpSampling1D(size=2), Cropping1D(cropping=(0, crop_dim)),
            Bidirectional(LSTM(LSTM_units, return_sequences=True, dropout=LSTM_dropout)),
            TimeDistributed(Dense(1, activation='tanh')),
            Flatten()
        ])
        
        # architecture of input space critic
        
        conv_filters = self.critic_x_params.get('conv_filters', 64)
        leakyrelu_slope = self.critic_x_params.get('leakyrelu_slope', 0.2)
        dropout = self.critic_x_params.get('dropout', 0.25)
        
        self.critic_x = Sequential([
            Reshape((self.input_dim, 1)),
            Conv1D(conv_filters, 5), LeakyReLU(alpha=leakyrelu_slope), Dropout(dropout),
            Conv1D(conv_filters, 5), LeakyReLU(alpha=leakyrelu_slope), Dropout(dropout),
            Conv1D(conv_filters, 5), LeakyReLU(alpha=leakyrelu_slope), Dropout(dropout),
            Conv1D(conv_filters, 5), LeakyReLU(alpha=leakyrelu_slope), Dropout(dropout),
            Flatten(),
            Dense(1)
        ])
        
        # architecture of latent space critic
        
        int_dim = self.critic_z_params.get('int_dim', 100)
        leakyrelu_slope = self.critic_z_params.get('leakyrelu_slope', 0.2)
        dropout = self.critic_z_params.get('dropout', 0.2)
        
        self.critic_z = Sequential([
            Dense(int_dim), LeakyReLU(alpha=leakyrelu_slope), Dropout(dropout),
            Dense(int_dim), LeakyReLU(alpha=leakyrelu_slope), Dropout(dropout),
            Dense(1)
        ])
        
        # training configuration of critics
        
        x = Input(shape=(self.input_dim,))
        z = Input(shape=(self.latent_dim,))
        
        self.encoder.trainable = False
        self.generator.trainable = False
        self.critic_x.trainable = True
        self.critic_z.trainable = True
        
        z_enc = self.encoder(x)
        x_gen = self.generator(z)
        
        score_x = self.critic_x(x)
        score_x_gen = self.critic_x(x_gen)
        
        x_mixed = _RandomWeightedMean()([x, x_gen])
        cx_grad_sq = _GradientSquared(self.critic_x)(x_mixed)
        
        self.critic_x_model = Model(inputs=[x, z], outputs=[score_x, score_x_gen, cx_grad_sq])
        self.critic_x_model.compile(loss=[_wasserstein_loss, _wasserstein_loss, 'mse'],
                                    loss_weights=[1, 1, 10],
                                    optimizer=self.optimizer)
        
        score_z = self.critic_z(z)
        score_z_enc = self.critic_z(z_enc)
        
        z_mixed = _RandomWeightedMean()([z, z_enc])
        cz_grad_sq = _GradientSquared(self.critic_z)(z_mixed)
        
        self.critic_z_model = Model(inputs=[x, z], outputs=[score_z, score_z_enc, cz_grad_sq])
        self.critic_z_model.compile(loss=[_wasserstein_loss, _wasserstein_loss, 'mse'],
                                    loss_weights=[1, 1, 10],
                                    optimizer=self.optimizer)
        
        # training configuration of encoder and generator
        
        self.encoder.trainable = True
        self.generator.trainable = True
        self.critic_x.trainable = False
        self.critic_z.trainable = False
        
        z_enc = self.encoder(x)
        x_gen = self.generator(z)
        x_recon = self.generator(z_enc)
        
        score_x_gen = self.critic_x(x_gen)
        score_z_enc = self.critic_z(z_enc)
        
        self.encoder_generator_model = Model(inputs=[x, z], 
                                             outputs=[score_x_gen, score_z_enc, x_recon])
        self.encoder_generator_model.compile(loss=[_wasserstein_loss, _wasserstein_loss, 'mse'],
                                             loss_weights=[1, 1, 10],
                                             optimizer=self.optimizer)
        
    def fit(self, x, batch_size=64, encoder_generator_freq=5, epochs=20):
        """Train the model with a collection of time series (x) and random
        vectors in the latent space sampled from a normal distribution.
        
        The time series (x) are expected to be a 2D-array, of the shape
        (# samples, # time stamps).
        
        In each epoch, the time series are shuffled randomly and divided into
        mini-batches. While every mini-batch is used to generate a training
        step for the critics, only every kth mini-batch is used to do so for
        the encoder and generator (k = encoder_generator_freq).
        """
        
        n_samples, input_dim = x.shape
        assert self.input_dim == input_dim

        self._build()
        
        n_batches = n_samples // batch_size
        ones = np.ones((batch_size, 1))
        
        for epoch in range(epochs):
            
            t0 = time.time()
            
            idx = np.random.permutation(n_samples)
            critic_x_losses, critic_z_losses, encoder_generator_losses = [], [], []
            
            for batch in range(n_batches):
                
                idx_batch = idx[batch_size * batch: batch_size * (batch + 1)]
                x_batch = x[idx_batch]
                z_batch = np.random.normal(size=(batch_size, self.latent_dim))
                
                # training step of each critic
                
                critic_x_losses.append(
                    self.critic_x_model.train_on_batch([x_batch, z_batch], [-ones, ones, ones])
                )
        
                critic_z_losses.append(
                    self.critic_z_model.train_on_batch([x_batch, z_batch], [-ones, ones, ones])
                )
                
                # training step of encoder + generator
                
                if (batch + 1) % encoder_generator_freq == 0:
                    encoder_generator_losses.append(
                        self.encoder_generator_model.train_on_batch(
                            [x_batch, z_batch], [-ones, -ones, x_batch]
                        )
                    )
            
            # Note: In each of the following arrays, the 0th element is the
            # total (weighted) loss, and the rest its respective components.
            
            critic_x_losses = np.mean(np.array(critic_x_losses), axis=0)
            critic_z_losses = np.mean(np.array(critic_z_losses), axis=0)
            encoder_generator_losses = np.mean(np.array(encoder_generator_losses), axis=0)
            
            t1 = time.time()
                    
            print(f'Epoch {epoch+1}/{epochs} ({t1-t0:.1f} secs)')
            print(f'  Critic X Loss: {critic_x_losses[0]:.6f} {critic_x_losses[1:]}')
            print(f'  Critic Z Loss: {critic_z_losses[0]:.6f} {critic_z_losses[1:]}')
            print(f'  Encoder Generator Loss: {encoder_generator_losses[0]:.6f} {encoder_generator_losses[1:]}')
    
    def predict(self, x):
        """Generate reconstruction of a collection of time series (x), via the
        encoder and the generator.

        The time series (x) are expected to be a 2D-array, of the shape
        (# samples, # time stamps).
         
        The outputs consist of the reconstructed time series (x_recon) as a 
        2D-array of the same shape, as well as the input space critic scores of
        the given time series (critic_score) as a 1D-array.        
        """
        x_recon = self.generator(self.encoder(x)).numpy()
        critic_score = self.critic_x(x).numpy()[:, 0]
        return x_recon, critic_score

