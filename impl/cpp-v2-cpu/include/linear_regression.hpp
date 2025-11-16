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
        std::size_t n_features);

    // predict
    std::vector<double> predict(
        const std::vector<double> &X,
        const std::size_t n_samples) const;

private:
    // hyperparameters
    std::size_t iterations_;
    double learning_rate_;

    // model parameters
    std::size_t n_features_;
    std::vector<double> weights_;
    double bias_;
    bool is_fitted_;

    // helpers
    void gradient_descent_step(
        const std::vector<double> &X,
        const std::vector<double> &y,
        std::size_t n_samples,
        std::size_t n_features,
        std::vector<double> &preds,
        std::vector<double> &errors,
        std::vector<double> &grads);

    void calculate_predictions(
        const std::vector<double> &X,
        std::size_t n_samples,
        std::size_t n_features,
        std::vector<double> &preds) const;

    double dot_row(
        const double *x_ptr,
        const double *w_ptr,
        std::size_t n_features) const;

    void check_shapes_fit(
        const std::vector<double> &X,
        const std::vector<double> &y,
        std::size_t n_samples,
        std::size_t n_features) const;

    void check_shapes_predict(
        const std::vector<double> &X,
        std::size_t n_samples,
        std::size_t n_features) const;
};
