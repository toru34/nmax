# jaxnm

PyTorch-like neural network module system in JAX.

The idea is borrowed from Parallax (https://github.com/srush/parallax. MIT License). The internal implementation is different.

## Instalation

```shell
pip clone https://github.com/toru34/jaxnm
cd jaxnm
python setup.py sdist bdist_wheel
pip install ./dist/jaxnm-0.0.1.tar.gz
```

## Example

See examples/xor/xor.py