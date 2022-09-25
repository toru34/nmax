import unittest

import jax
import jax.nn as nn
import jax.numpy as jnp

from nmax import Module, Parameter, Constant


class Dense(Module):

    W: Parameter
    b: Parameter

    def __init__(self, key, in_dim, out_dim):
        self.W = jax.random.normal(key, (in_dim, out_dim))
        self.b = jnp.zeros(out_dim)

    def forward(self, x):
        return x @ self.W + self.b


class MLP(Module):

    dense1: Module
    dense2: Module

    def __init__(self, key, in_dim, hid_dim, out_dim):
        keys = jax.random.split(key)

        self.dense1 = Dense(keys[0], in_dim, hid_dim)
        self.dense2 = Dense(keys[1], hid_dim, out_dim)

    def forward(self, x):
        x = jnp.tanh(self.dense1(x))
        return nn.sigmoid(self.dense2(x))


class TestXOR(unittest.TestCase):
    def test_xor(self):
        def loss_fn(model, x, y):
            p = model(x)
            loss = - y * jnp.log(p) - (1 - y) * jnp.log(1 - p)
            return loss.mean()
        
        def sgd(param, grad, lr=4.0):
            return param - lr * grad

        x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = jnp.array([[0], [1], [1], [0]])

        model = MLP(
            key=jax.random.PRNGKey(34),
            in_dim=2,
            hid_dim=2,
            out_dim=1
        )

        grad_fn = jax.jit(jax.grad(loss_fn))

        for epoch in range(40):
            grads = grad_fn(model, x, y)

            model = jax.tree_multimap(sgd, model, grads)

            if (epoch + 1) % 10 == 0:
                loss = loss_fn(model, x, y)
        
        loss = loss_fn(model, x, y)
        preds = model(x)
        
        self.assertTrue(preds[0] < 0.1)
        self.assertTrue(preds[1] > 0.9)
        self.assertTrue(preds[2] > 0.9)
        self.assertTrue(preds[3] < 0.1)


if __name__ == '__main__':
    unittest.main()