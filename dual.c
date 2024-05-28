/**
* @file    dual.c
* @brief   This file contains primitive operations for performing 
*          Automated-Differentiation for scalar function.
*
* @authors Abhijeet Gangan <abhijeetgangan@g.ucla.edu>
*/

#include "dual.h"
#include <stdlib.h>
#include <math.h>

// Addition rule for dual numbers
dual add_dual(dual f, dual g) {
    dual result;
    result.value = f.value + g.value;
    result.derivative = f.derivative + g.derivative;
    return result;
};

dual add_const(dual f, double c) {
    dual result;
    result.value = f.value + c;
    result.derivative = f.derivative;
    return result;
};

// Subtraction rule for dual numbers
dual sub_dual(dual f, dual g) {
    dual result;
    result.value = f.value - g.value;
    result.derivative = f.derivative - g.derivative;
    return result;
};

// Product rule for dual numbers
dual prd_dual(dual f, dual g) {
    dual result;
    result.value = f.value * g.value;
    result.derivative = f.derivative * g.value + g.derivative * f.value;
    return result;
};

dual prd_const(dual f, double c) {
    dual result;
    result.value = f.value * c;
    result.derivative = f.derivative * c;
    return result;
};

// Quotient rule for dual numbers
dual quo_dual(dual f, dual g) {
    dual result;
    result.value = f.value / g.value;
    result.derivative = (f.derivative * g.value - g.derivative * f.value) / (g.value * g.value);
    return result;
};

dual quo_const(double c, dual f) {
    dual result;
    result.value = c / f.value;
    result.derivative = 1 / f.derivative;
    return result;
};

// Power rule
dual pow_by_repeated_squaring(dual x, int n) {
    if (n < 0) {
        x = quo_const(1, x);
    }
    
    long num = labs(n);   
    dual pow;
    pow.value = 1;
    pow.derivative = 0;
        
    while (num) {
        if(num & 1) {
            pow = prd_dual(pow, x);
        }
        x = prd_dual(x, x);
        num >>= 1;
    }
    return pow;
};

dual pow_const(dual f, int n) {
    return pow_by_repeated_squaring(f, n);
};

// Exponential
dual exp_dual(dual f) {
    dual result;
    result.value = exp(f.value);
    result.derivative = exp(f.value) * f.derivative;
    return result;
};

// Sin
dual sin_dual(dual f) {
    dual result;
    result.value = sin(f.value);
    result.derivative = cos(f.value) * f.derivative;
    return result;
};

// Cos
dual cos_dual(dual f) {
    dual result;
    result.value = cos(f.value);
    result.derivative = -sin(f.value) * f.derivative;
    return result;
};

// Tan
dual tan_dual(dual f) {
    dual result;
    result.value = tan(f.value);
    result.derivative = f.derivative / (cos(f.value) * cos(f.value));
    return result;
};