#pragma once

#include <vector>
#include <cstddef>

class LinearRegression
{
public:
    // Constructor
    LinearRegression(std::size_t iterations, double learning_rate);

    // fit method
    void fit(
        const std::vector<double> &X,
        const std::vector<double> &y,
        std::size_t n_samples,
        std::size_t features);

    // predict
    std::vector<double> predict(
        const std::vector<double> &X,
        const std::size_t n_samples,
        const std::size_t n_features);

private:
    // hyperparameters
    std::size_t iterations_;
    double learning_rate_;

    // model parameters
    std::size_t n_features_;
    std::vector<double> weights_;
    double bias_;

    // helpers
    void gradient_descent_step(
        const std::vector<double> &X,
        const std::vector<double> &y,
        std::size_t n_samples,
        std::size_t n_features,
        std::vector<double> &preds,
        std::vector<double> &errors,
        std::vector<double> &grads);

    double dotRow(
        const double *x_ptr,
        const double *w_ptr,
        std::size_t n_features) const;
};