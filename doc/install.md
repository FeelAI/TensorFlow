# Install

## Install TensorFlow on Ubuntu

- TensorFlow with CPU support only
- TensorFlow with GPU support

## Install with native pip

```shell
$ pip install tensorflow
```

> HTTPS TimeOut with the help of ShadowSocks

## Validate installation

```shell
$ python

# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
session = tf.Session()
print(session.run(hello))
```

