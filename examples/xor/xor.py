import jax
import jax.nn as nn
import jax.numpy as jnp

from nmax import Module, Parameter, Constant


def main():
    def loss_fn(model, x, y):
        p = model(x)
        loss = - y * jnp.log(p) - (1 - y) * jnp.log(1 - p)
        return loss.mean()
    

    def sgd(param, grad, lr=4.0):
        return param - lr * grad
    

    x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = jnp.array([[0], [1], [1], [0]])

    model = MLP(
        in_dim=2,
        hid_dim=2,
        out_dim=1,
        key=jax.random.PRNGKey(34),
    )

    grad_fn = jax.jit(jax.grad(loss_fn))

    for epoch in range(40):
        grads = grad_fn(model, x, y)

        model = jax.tree_multimap(sgd, model, grads)

        if (epoch + 1) % 10 == 0:
            loss = loss_fn(model, x, y)
            print(loss.item())
    
    print(model(x))


class Dense(Module):

    W: Parameter
    b: Parameter

    def __init__(self, in_dim, out_dim, key):
        self.W = jax.random.normal(key, (in_dim, out_dim))
        self.b = jnp.zeros(out_dim)
    
    def forward(self, x):
        return x @ self.W + self.b


class MLP(Module):

    dense1: Module
    dense2: Module

    def __init__(self, in_dim, hid_dim, out_dim, key):
        keys = jax.random.split(key)

        self.dense1 = Dense(in_dim, hid_dim, keys[0])
        self.dense2 = Dense(hid_dim, out_dim, keys[1])
    
    def forward(self, x):
        x = jnp.tanh(self.dense1(x))
        return nn.sigmoid(self.dense2(x))

if __name__ == '__main__':
    main()