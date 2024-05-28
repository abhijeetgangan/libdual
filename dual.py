import ctypes
from ctypes import c_double, c_int
from typing import Callable, Any, Tuple

# Define the dual struct
class dual(ctypes.Structure):
    _fields_ = [("value", c_double), ("derivative", c_double)]

# Load the shared library
libdual = ctypes.CDLL('./libdual.so')

# Define argument and return types for the C functions
# This are defined in the dual.c file
libdual.add_dual.argtypes = (dual, dual)
libdual.add_dual.restype = dual

libdual.add_const.argtypes = (dual, c_double)
libdual.add_const.restype = dual

libdual.sub_dual.argtypes = (dual, dual)
libdual.sub_dual.restype = dual

libdual.prd_dual.argtypes = (dual, dual)
libdual.prd_dual.restype = dual

libdual.prd_const.argtypes = (dual, c_double)
libdual.prd_const.restype = dual

libdual.quo_dual.argtypes = (dual, dual)
libdual.quo_dual.restype = dual

libdual.quo_const.argtypes = (c_double, dual)
libdual.quo_const.restype = dual

libdual.pow_by_repeated_squaring.argtypes = (dual, c_int)
libdual.pow_by_repeated_squaring.restype = dual

libdual.pow_const.argtypes = (dual, c_int)
libdual.pow_const.restype = dual

libdual.exp_dual.argtypes = (dual,)
libdual.exp_dual.restype = dual

libdual.sin_dual.argtypes = (dual,)
libdual.sin_dual.restype = dual

libdual.cos_dual.argtypes = (dual,)
libdual.cos_dual.restype = dual

libdual.tan_dual.argtypes = (dual,)
libdual.tan_dual.restype = dual


# CDual class performes operations on dual numbers
class CDual:
    def __init__(self, value, derivative):
        self.dual = dual(value, derivative)

    def __add__(self, other):
        if isinstance(other, CDual):
            result = libdual.add_dual(self.dual, other.dual)
        else:  
            # The other number is a constant
            result = libdual.add_const(self.dual, other)
        return CDual(result.value, result.derivative)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CDual):
            result = libdual.sub_dual(self.dual, other.dual)
        else:
            # The other number is a constant
            result = libdual.add_const(self.dual, -other)
        return CDual(result.value, result.derivative)

    def __rsub__(self, other):
        if isinstance(other, CDual):
            result = libdual.sub_dual(other.dual, self.dual)
        else:
            # The other number is a constant
            result = libdual.add_const_1(dual(other, 0), -self.dual.value)
        return CDual(result.value, result.derivative)

    def __mul__(self, other):
        if isinstance(other, CDual):
            result = libdual.prd_dual(self.dual, other.dual)
        else:
            # The other number is a constant
            result = libdual.prd_const(self.dual, other)
        return CDual(result.value, result.derivative)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, CDual):
            result = libdual.quo_dual(self.dual, other.dual)
        else:
            # The other number is a constant
            result = libdual.prd_dual(self.dual, dual(1 / other, 0))
        return CDual(result.value, result.derivative)

    def __rtruediv__(self, other):
        if isinstance(other, CDual):
            result = libdual.quo_dual(other.dual, self.dual)
        else:
            # The other number is a constant
            result = libdual.quo_const(other, self.dual)
        return CDual(result.value, result.derivative)

    def __pow__(self, n):
        result = libdual.pow_const(self.dual, n)
        return CDual(result.value, result.derivative)

    def exp(self):
        result = libdual.exp_dual(self.dual)
        return CDual(result.value, result.derivative)

    def sin(self):
        result = libdual.sin_dual(self.dual)
        return CDual(result.value, result.derivative)

    def cos(self):
        result = libdual.cos_dual(self.dual)
        return CDual(result.value, result.derivative)

    def tan(self):
        result = libdual.tan_dual(self.dual)
        return CDual(result.value, result.derivative)

    def grad(function: Callable[[float], Any], value: float) -> float:
        """
        Returns the gradient at a value
        """
        x = CDual(value, 1)
        return function(x).dual.derivative
    
    def value_and_grad(function: Callable[[float], Any], input: float) -> Tuple[float, float]:
        """
        Returns the function and gradient at a value
        """
        x = CDual(input, 1)
        return function(x).dual.value, function(x).dual.derivative
    
    def __repr__(self):
        return f"CDualNumber(value={self.dual.value}, derivative={self.dual.derivative})"