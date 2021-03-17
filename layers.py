import numpy as np
import tensorflow as tf


# ODE function
class ConcatSquash(tf.keras.layers.Layer):
    def __init__(self, use_swish=True, nhidden=512):
        super(ConcatSquash, self).__init__()
        self.dense = tf.keras.layers.Dense(nhidden)
        self._hyper_bias = tf.keras.layers.Dense(nhidden, use_bias=False)
        self._hyper_gate = tf.keras.layers.Dense(nhidden, activation='sigmoid')
        if use_swish:
            self._idnet = tf.keras.layers.Lambda(lambda x: tf.nn.silu(x))
        else:
            self._idnet = tf.keras.layers.Lambda(lambda x: x)

    def call(self, t, x, cond):
        x = self.dense(x) * self._hyper_gate(cond) + self._hyper_bias(cond)
        return self._idnet(x)

class ODEFnc(tf.keras.layers.Layer):
    def __init__(self, nhidden=512, stack=4):
        super(ODEFnc, self).__init__()
        self.concatsquash = [ConcatSquash(nhidden=nhidden) for i in range(stack - 1)] + [ConcatSquash(use_swish=False,nhidden=nhidden)]
    
    def call(self, t, x, cond):
        t = tf.broadcast_to(t, tf.shape(cond))
        cond = tf.concat([t, cond], axis=-1)
        for lyr in self.concatsquash:
            x = lyr(t, x, cond)
        return x
