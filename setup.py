import setuptools

setuptools.setup(
    name='nmax',
    version='0.0.1',
    author='Toru Fujino',
    author_email='toru.fb34@gmail.com',
    description='Jax\'s neural network module system',
    long_description=open('README.md', 'r').read(),
    url='https://github.com/toru34/nmax',
    packages=setuptools.find_packages(),
    python_requires='>=3.10'
)