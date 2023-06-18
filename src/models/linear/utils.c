#include "utils.h"
#include <math.h>
#include <stddef.h>

double MSE(const double* y_test, const double* pred, size_t size) {
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = y_test[i] - pred[i];
        sum += diff * diff;
    }
    return sum / size;
}

double R2(const double* y_test, const double* pred, size_t size) {
    double mean = gsl_stats_mean(y_test, 1, size);

    double ss_total = 0.0;
    double ss_residual = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff_total = y_test[i] - mean;
        double diff_residual = y_test[i] - pred[i];
        ss_total += diff_total * diff_total;
        ss_residual += diff_residual * diff_residual;
    }
    return 1.0 - (ss_residual / ss_total);
}
