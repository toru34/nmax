import jax
import jax.nn as nn
import jax.numpy as jnp

from jaxnm import BaseModule


class Dense(BaseModule):
    def __init__(self, key, in_dim, out_dim):
        super().__init__()

        self.W = jax.random.normal(key, (in_dim, out_dim))
        self.b = jnp.zeros(out_dim)
    
    def forward(self, x):
        return x @ self.W + self.b


class MLP(BaseModule):
    def __init__(self, key, in_dim, hid_dim, out_dim):
        super().__init__()

        keys = jax.random.split(key)

        self.dense1 = Dense(keys[0], in_dim, hid_dim)
        self.dense2 = Dense(keys[1], hid_dim, out_dim)
    
    def forward(self, x):
        x = jnp.tanh(self.dense1(x))
        return nn.sigmoid(self.dense2(x))


def loss_fn(model, x, y):
    y_pred = model(x)
    loss = - y * jnp.log(y_pred) - (1 - y) * jnp.log(1 - y_pred)
    return loss.mean()


def sgd(param, grad, lr=1.0):
    return param - lr * grad


def main():
    key = jax.random.PRNGKey(34)

    train_x = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_y = jnp.array([[0], [1], [1], [0]])

    test_x = train_x
    test_y = train_y

    n_epochs = 200

    model = MLP(key, in_dim=2, hid_dim=2, out_dim=1)
    grad_fn = jax.jit(jax.grad(loss_fn))

    for epoch in range(n_epochs):
        grads = grad_fn(model, train_x, train_y)

        model = jax.tree_multimap(sgd, model, grads)

        if (epoch + 1) % (n_epochs // 10) == 0:
            loss = loss_fn(model, train_x, train_y)
            print(loss.item())
    
    y_pred = model(test_x)
    print(y_pred)


if __name__ == '__main__':
    main()