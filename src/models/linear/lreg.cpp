#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <stdexcept>   
#include <armadillo>

using namespace std;

double MSE(const arma::vec& y_test, const arma::vec& pred) {
    arma::vec diff = y_test - pred;
    return arma::mean(diff % diff); 
}

double R2(const arma::vec& y_test, const arma::vec& pred) {
    double mean = arma::mean(y_test);
    arma::vec diff_total = y_test - mean;
    arma::vec diff_residual = y_test - pred;
    double ss_total = arma::dot(diff_total, diff_total);
    double ss_residual = arma::dot(diff_residual, diff_residual);
    return 1.0 - (ss_residual / ss_total);
}

class LinearRegression {
    private:
        double rate;
        int epochs;
        arma::vec w;
        double b;

    public:
        LinearRegression(double rate = 0.001, int epochs = 1000) : rate(rate), epochs(epochs), b(0.0) {}

    void fit(const arma::mat& X, const arma::vec& y) {
        int samples = X.n_rows, features = X.n_cols;
        w.zeros(features);
        b = 0;

        for (int i = 0; i < epochs; ++i) {
            arma::vec y_hat = X * w + b;
            w -= rate * (2.0 / samples) * (X.t() * (y_hat - y));
            b -= rate * (2.0 / samples) * arma::sum(y_hat - y);
        }
    } 

    void fit_mini_batch(const arma::mat& X, const arma::vec& y, int batch_size = 32, float tolerance = 1e-5) {
        int samples = X.n_rows, features = X.n_cols;
        arma::vec w = arma::zeros<arma::vec>(features);
        double b = 0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            arma::uvec idx = arma::shuffle(arma::linspace<arma::uvec>(0, samples - 1, samples));
            arma::mat X_shuffled = X.rows(idx);
            arma::vec y_shuffled = y.elem(idx);

            double mse_prev = std::numeric_limits<double>::infinity();
            for (int i = 0; i < samples; i += batch_size) {
                arma::mat X_i = X_shuffled.rows(i, std::min(i + batch_size, samples) - 1);
                arma::vec y_i = y_shuffled.subvec(i, std::min(i + batch_size, samples) - 1);
                arma::vec y_hat = X_i * w + b;
                w -= rate * (2.0 / samples) * X_i.t() * (y_hat - y_i);
                b -= rate * (2.0 / samples) * arma::sum(y_hat - y_i);
    
                arma::vec y_hat_all = X * w + b;
                double mse_curr = MSE(y, y_hat_all);
    
                if (mse_prev - mse_curr < tolerance) {
                    return;
                }
    
                mse_prev = mse_curr;
            }
        }
    }

    arma::mat predict(const arma::mat& X) {
        return X * w + b;
    }
};

//extern "C" {
//    LinearRegression* LinearRegression_new(double rate, int epochs) {
//        return new LinearRegression(rate, epochs);
//    }
//    
//    void LinearRegression_fit(LinearRegression* lr, double* X, double* y, int samples, int features) {
//        arma::mat X_mat(X, samples, features);
//        arma::vec y_vec(y, samples);
//        lr->fit(X_mat, y_vec);
//    }
//
//    void LinearRegression_fit_mini_batch(LinearRegression* lr, double* X, double* y, int samples, int features, int batch_size, float tolerance) {
//        arma::mat X_mat(X, samples, features);
//        arma::vec y_vec(y, samples);
//        lr->fit_mini_batch(X_mat, y_vec, batch_size, tolerance);
//    }
//
//    double* LinearRegression_predict(LinearRegression* lr, double* X, int samples, int features) {
//        arma::mat X_mat(X, samples, features);
//        arma::vec y_hat = lr->predict(X_mat);
//        return y_hat.memptr();
//    }
//
//    void LinearRegression_delete(LinearRegression* lr) {
//        delete lr;
//    }
//}
