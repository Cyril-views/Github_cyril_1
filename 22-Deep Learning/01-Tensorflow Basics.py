"""
___

<a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
___
# TensorFlow Basics

Remember to reference the video for full explanations, this is just a notebook for code reference.

You can import the library:
"""

import tensorflow as tf

print(tf.__version__)

"""
### Simple Constants

Let's show how to create a simple constant with Tensorflow, which TF stores as a tensor object:
"""

hello = tf.constant('Hello World')

type(hello)

x = tf.constant(100)

type(x)

"""
### Running Sessions

Now you can create a TensorFlow Session, which is a class for running TensorFlow operations.

A `Session` object encapsulates the environment in which `Operation`
objects are executed, and `Tensor` objects are evaluated. For example:
"""

sess = tf.Session()

sess.run(hello)

type(sess.run(hello))

sess.run(x)

type(sess.run(x))

"""
## Operations

You can line up multiple Tensorflow operations in to be run during a session:
"""

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(x+y))
    print('Subtraction',sess.run(x-y))
    print('Multiplication',sess.run(x*y))
    print('Division',sess.run(x/y))

"""
### Placeholder

You may not always have the constants right away, and you may be waiting for a constant to appear after a cycle of operations. **tf.placeholder** is a tool for this. It inserts a placeholder for a tensor that will be always fed.

**Important**: This tensor will produce an error if evaluated. Its value must be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`. For example, for a placeholder of a matrix of floating point numbers:

    x = tf.placeholder(tf.float32, shape=(1024, 1024))

Here is an example for integer placeholders:
"""

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

x

type(x)

"""
### Defining Operations
"""

add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)

"""
Running operations with variable input:
"""

d = {x:20,y:30}

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(add,feed_dict=d))
    print('Subtraction',sess.run(sub,feed_dict=d))
    print('Multiplication',sess.run(mul,feed_dict=d))

"""
Now let's see an example of a more complex operation, using Matrix Multiplication. First we need to create the matrices:
"""

import numpy as np
# Make sure to use floats here, int64 will cause an error.
a = np.array([[5.0,5.0]])
b = np.array([[2.0],[2.0]])

a

a.shape

b

b.shape

mat1 = tf.constant(a)

mat2 = tf.constant(b)

"""
The matrix multiplication operation:
"""

matrix_multi = tf.matmul(mat1,mat2)

"""
Now run the session to perform the Operation:
"""

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)

"""
That is all for now! Next we will expand these basic concepts to construct out own Multi-Layer Perceptron model!
"""

"""
# Great Job!
"""
