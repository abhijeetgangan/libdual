/**
* @file    dual.h
* @brief   This file declares primitive operations for performing 
*          Automated-Differentiation for scalar function.
*
* @authors Abhijeet Gangan <abhijeetgangan@g.ucla.edu>
*/

#ifndef DUAL_H
#define DUAL_H

// Define a struct for dual type
typedef struct {
    double value;
    double derivative;
} dual;

dual add_dual(dual f, dual g);
dual add_const(dual f, double c);
dual sub_dual(dual f, dual g);
dual prd_dual(dual f, dual g);
dual prd_const(dual f, double c);
dual quo_dual(dual f, dual g);
dual quo_const(double c, dual f);
dual pow_by_repeated_squaring(dual x, int n);
dual pow_const(dual f, int n);
dual exp_dual(dual f);
dual sin_dual(dual f);
dual cos_dual(dual f);
dual tan_dual(dual f);

#endif // DUAL_H