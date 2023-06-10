#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

double MSE(const double* y_test, const double* pred, size_t size);
double R2(const double* y_test, const double* pred, size_t size);

#endif
