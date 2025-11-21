#pragma once

#include <vector>
#include <cstddef>

class LinearRegressionV22
{
public:
    // Constructor
    using scalar_t = float;
    LinearRegressionV22(std::size_t iterations, double learning_rate);

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
    scalar_t learning_rate_;

    // model parameters
    std::size_t n_features_;
    std::vector<scalar_t> weights_;
    scalar_t bias_;
    bool is_fitted_;

    // helpers
    void gradient_descent_step(
        const std::vector<scalar_t> &X,
        const std::vector<scalar_t> &y,
        std::size_t n_samples,
        std::size_t n_features,
        std::vector<scalar_t> &preds,
        std::vector<scalar_t> &errors,
        std::vector<scalar_t> &grads);

    void calculate_predictions(
        const std::vector<scalar_t> &X,
        std::size_t n_samples,
        std::size_t n_features,
        std::vector<scalar_t> &preds) const;

    scalar_t dot_row(
        const scalar_t *a_ptr,
        const scalar_t *b_ptr,
        std::size_t n_features) const;

    void check_shapes_fit(
        const std::vector<scalar_t> &X,
        const std::vector<scalar_t> &y,
        std::size_t n_samples,
        std::size_t n_features) const;

    void check_shapes_predict(
        const std::vector<scalar_t> &X,
        std::size_t n_samples,
        std::size_t n_features) const;
};
