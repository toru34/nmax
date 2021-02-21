import jax
import jax.nn as nn
import jax.numpy as jnp

from jaxnm import Module, Parameter, Constant


def main():
    def loss_fn(model, x, y):
        p = model(x)
        loss = - y * jnp.log(p) - (1 - y) * jnp.log(1 - p)
        return loss.mean()
    

    def sgd(param, grad, lr=1.0):
        return param - lr * grad
    

    x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = jnp.array([[0], [1], [1], [0]])

    model = MLP(
        in_dim=2,
        hid_dim=2,
        out_dim=1,
        rng=jax.random.PRNGKey(34),
    )

    for epoch in range(100):
        grads = grad_fn(model, x, y)

        model = jax.tree_multimap(sgd, model, grads)

        if (epoch + 1) % 10 == 0:
            loss = loss_fn(model, x, y)
            print(loss.item())
    
    print(model(x))


class Dense(Module):

    W: Parameter
    b: Parameter

    def __init__(self, in_dim, out_dim, rng):
        self.W = jax.random.normal(rng, (in_dim, out_dim))
        self.b = jnp.zeros(out_dim)


class MLP(Module):

    dense1: Module
    dense2: Module

    def __init__(self, in_dim, hid_dim, out_dim, rng):
        rngs = jax.random.split(rng)

        self.dense1 = Dense(in_dim, hid_dim, rngs[0])
        self.dense2 = Dense(hid_dim, out_dim, rngs[1])
    
    def forward(self, x):
        x = jnp.tanh(self.dense1(x))
        return nn.sigmoid(x @ self.W + self.b)

if __name__ == '__main__':
    main()