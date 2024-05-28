from dual import CDual
from functools import partial

# Define a function to take the derivative
def fn(x : float):
    return x * x

grad_fn = partial(CDual.grad, fn)
value_and_grad_fn = partial(CDual.value_and_grad, fn)

x = 3.0
print(f"Gradient of x^2 at {x} is {grad_fn(x)}")
print(f"Value of x^2 at {x} is {value_and_grad_fn(x)[0]} and gradient is {value_and_grad_fn(x)[1]}")

# Define a function to take the derivative
def gn(x : float):
    return CDual.sin(x)

x = 0.0
grad_gn = partial(CDual.grad, gn)
value_and_grad_gn = partial(CDual.value_and_grad, gn)

print(f"Gradient of sin(x) at {x} is {grad_gn(x)}")
print(f"Value of sin(x) at {x} is {value_and_grad_gn(x)[0]} and gradient is {value_and_grad_gn(x)[1]}")