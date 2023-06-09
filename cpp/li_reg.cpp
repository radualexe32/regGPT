#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <stdexcept>   

using namespace std;

double MSE(vector<double>& y_test, vector<double>& pred) {
    double sum = 0.0;
    int n = y_test.size();
    for (int i = 0; i < n; i++) {
        double diff = y_test[i] - pred[i];
        sum += diff * diff;
    }
    return sum / n;
}

double R2(vector<double>& y_test, vector<double>& pred) {
    double mean = accumulate(y_test.begin(), y_test.end(), 0.0) / y_test.size();
    double ss_total = 0.0;
    double ss_residual = 0.0;
    int n = y_test.size();
    for (int i = 0; i < n; i++) {
        double diff_total = y_test[i] - mean;
        double diff_residual = y_test[i] - pred[i];
        ss_total += diff_total * diff_total;
        ss_residual += diff_residual * diff_residual;
    }
    return 1.0 - (ss_residual / ss_total);
}

class LinearRegression {
    private:
        double rate;
        int epochs;
        vector<double> w;
        double b;
        vector<vector<double>> w_hist;
        vector<double> b_hist;
        vector<double> mse_hist;
        vector<double> r2_hist;
        vector<double> time_hist;

    public:
        LinearRegression(double rate = 0.001, int epochs = 1000) : rate(rate), epochs(epochs), b(0.0) {}

    void fit(const std::vector<double>& X, const std::vector<double>& y) {
        throw std::runtime_error("Not implemented");
    } 

    void fit_mini_batch(const std::vector<double>& X, const std::vector<double>& y, int batch_size = 32, float tolerance = 1e-5) {
        throw std::runtime_error("Not implemented");
    }

    std::vector<double> predict(const std::vector<double>& X) {
        throw std::runtime_error("Not implemented");
    }
};