### Implementing autograd with dual numbers (Only for learning purpose).

It should work without any dependencies.

To start, first build a shared object as follows:
```shell
gcc -shared -o libdual.so -fPIC dual.c 
```

Then run the example using:
```shell
python example.py
```

### Usage:
The [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function and it's derivative is defined as:

$g(x) = \frac{e^{x}}{1 + e^{x}}$

$g^{'}(x) = g(x)(1-g(x))$

Let's verify the numerical values using AD.
```python 
# First import CDual
from dual import CDual
from functools import partial
import math

# Define the sigmoid function
def gx(x: float):
    return math.exp(x) / (1 + math.exp(x))

def gx_dual(x : float):
    return CDual.exp(x) / (1 + CDual.exp(x))

# Expected derivative of sigmoid
def grad_gx(x: float):
    return gx(x) * (1 - gx(x))

# This function returns the value and derivative at a given point
value_and_grad_fn = partial(CDual.value_and_grad, gx_dual)

# At 0 we should get a value of 0.5 and derivative of 0.25
x = 0.0
print(f"The expected value of sigmoid(x) at {x} is {gx(x)} and gradient is {grad_gx(x)}")
print(f"Value of sigmoid(x) at {x} using AD is {value_and_grad_fn(x)[0]} and gradient is {value_and_grad_fn(x)[1]}")
```