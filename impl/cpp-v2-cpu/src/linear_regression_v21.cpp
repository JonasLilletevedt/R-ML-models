#include "linear_regression_v21.hpp"
#include <stdexcept>

LinearRegressionV21::LinearRegressionV21(std::size_t iterations, double learning_rate) : iterations_(iterations),
                                                                                         learning_rate_(learning_rate),
                                                                                         n_features_(0),
                                                                                         weights_(),
                                                                                         bias_(0.0),
                                                                                         is_fitted_(false)
{
}

void LinearRegressionV21::fit(const std::vector<double> &X, const std::vector<double> &y, std::size_t n_samples, std::size_t n_features)
{
    // check shapes
    check_shapes_fit(X, y, n_samples, n_features);

    LinearRegressionV21::n_features_ = n_features;
    LinearRegressionV21::weights_.assign(n_features_, 0.0);
    LinearRegressionV21::bias_ = 0.0;

    std::vector<double> preds(n_samples, 0.0);
    std::vector<double> errors(n_samples, 0.0);
    std::vector<double> grads(n_features_, 0.0);

    for (std::size_t i = 0; i < iterations_; i++)
    {
        gradient_descent_step(
            X,
            y,
            n_samples,
            n_features_,
            preds,
            errors,
            grads);
    }

    is_fitted_ = true;
}

std::vector<double> LinearRegressionV21::predict(const std::vector<double> &X, const std::size_t n_samples) const
{
    // Check fitted
    if (!is_fitted_)
    {
        throw std::logic_error("Model is not fitted. Call fit() before predict()");
    }

    // Check shapes
    check_shapes_predict(X, n_samples, n_features_);

    std::vector<double> preds(n_samples, 0.0);
    // predictions: p = Xw + b
    calculate_predictions(X, n_samples, n_features_, preds);

    return preds;
}

void LinearRegressionV21::gradient_descent_step(
    const std::vector<double> &X,
    const std::vector<double> &y,
    std::size_t n_samples,
    std::size_t n_features,
    std::vector<double> &preds,
    std::vector<double> &errors,
    std::vector<double> &grads)
{
    const double *X_ptr = X.data();
    const double *y_ptr = y.data();
    double *w_ptr = weights_.data();
    double *p_ptr = preds.data();
    double *e_ptr = errors.data();
    double *g_ptr = grads.data();
    double bias_gradient = 0;

    // predictions: p = Xw + b
    calculate_predictions(X, n_samples, n_features, preds);

// errors: e = p - y
#pragma omp parallel for
    for (std::size_t i = 0; i < n_samples; i++)
    {
        e_ptr[i] = p_ptr[i] - y_ptr[i];
    }

    // gradients: g = X^T * e / n_samples

    // start from zero
    std::fill(grads.begin(), grads.end(), 0.0);

#pragma omp parallel
    {
        std::vector<double> g_loc(n_features, 0.0);
        double *g_loc_ptr = g_loc.data();

#pragma omp for nowait schedule(static)
        for (std::size_t i = 0; i < n_samples; i++)
        {
            for (std::size_t j = 0; j < n_features; j++)
            {
                g_loc_ptr[j] += X_ptr[i * n_features + j] * e_ptr[i];
            }
        }
#pragma omp critical
        {
            for (std::size_t j = 0; j < n_features; j++)
            {
                g_ptr[j] += g_loc_ptr[j];
            }
        }
    }
    // Divide sum by m to get average g
    for (std::size_t j = 0; j < n_features; j++)
    {
        g_ptr[j] = g_ptr[j] / n_samples;
    }

    // bias
    for (std::size_t i = 0; i < n_samples; i++)
    {
        bias_gradient += e_ptr[i];
    }
    bias_gradient /= n_samples;

    // update weights
#pragma omp parallel for
    for (std::size_t j = 0; j < n_features; j++)
    {
        w_ptr[j] -= learning_rate_ * g_ptr[j];
    }

    // update bias
    bias_ -= learning_rate_ * bias_gradient;
}

void LinearRegressionV21::calculate_predictions(const std::vector<double> &X, std::size_t n_samples, std::size_t n_features, std::vector<double> &preds) const
{
    const double *X_ptr = X.data();
    const double *w_ptr = weights_.data();
    double *p_ptr = preds.data();
#pragma omp parallel for
    for (std::size_t i = 0; i < n_samples; i++)
    {
        const double *row_ptr = X_ptr + i * n_features;
        p_ptr[i] = dot_row(row_ptr, w_ptr, n_features) + bias_;
    }
}

double LinearRegressionV21::dot_row(const double *a_ptr, const double *b_ptr, std::size_t n) const
{
    double dotProduct(0.0);

    for (std::size_t i = 0; i < n; i++)
    {
        dotProduct += a_ptr[i] * b_ptr[i];
    }

    return dotProduct;
}

void LinearRegressionV21::check_shapes_fit(const std::vector<double> &X, const std::vector<double> &y, std::size_t n_samples, std::size_t n_features) const
{
    if (n_samples <= 0)
    {
        throw std::invalid_argument("n_samples must be greater than zero");
    }
    if (n_features <= 0)
    {
        throw std::invalid_argument("n_features must be greater than zero");
    }
    if (X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X size does not match n_samples * n_features");
    }
    if (y.size() != n_samples)
    {
        throw std::invalid_argument("y size does not match n_samples");
    }
}

void LinearRegressionV21::check_shapes_predict(const std::vector<double> &X, std::size_t n_samples, std::size_t n_features)
    const
{
    if (n_samples <= 0)
    {
        throw std::invalid_argument("n_samples must be greater than zero");
    }
    if (n_features <= 0)
    {
        throw std::invalid_argument("n_features must be greater than zero");
    }
    if (X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X size does not match n_samples * n_features");
    }
}
