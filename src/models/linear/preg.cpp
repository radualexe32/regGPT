#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

class PolynomialRegression {
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
        PolynomialRegression(double rate = 0.001, int epochs = 1000) : rate(rate), epochs(epochs), b(0.0) {}

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