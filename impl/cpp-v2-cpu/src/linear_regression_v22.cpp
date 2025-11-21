#include "linear_regression_v22.hpp"
#include <stdexcept>

LinearRegressionV22::LinearRegressionV22(std::size_t iterations, double learning_rate)
    : iterations_(iterations),
      learning_rate_(static_cast<scalar_t>(learning_rate)),
      n_features_(0),
      weights_(),
      bias_(scalar_t{0}),
      is_fitted_(false)
{
}

void LinearRegressionV22::fit(const std::vector<double> &X_inp,
                              const std::vector<double> &y_inp,
                              std::size_t n_samples,
                              std::size_t n_features)
{
    // Convert input to float once
    const std::size_t total = n_samples * n_features;
    std::vector<scalar_t> X(total);
    std::vector<scalar_t> y(n_samples);

    for (std::size_t i = 0; i < total; ++i)
    {
        X[i] = static_cast<scalar_t>(X_inp[i]);
    }
    for (std::size_t i = 0; i < n_samples; ++i)
    {
        y[i] = static_cast<scalar_t>(y_inp[i]);
    }

    // check shapes
    check_shapes_fit(X, y, n_samples, n_features);

    n_features_ = n_features;
    weights_.assign(n_features_, scalar_t{0});
    bias_ = scalar_t{0};

    std::vector<scalar_t> preds(n_samples, scalar_t{0});
    std::vector<scalar_t> errors(n_samples, scalar_t{0});
    std::vector<scalar_t> grads(n_features_, scalar_t{0});

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

std::vector<double> LinearRegressionV22::predict(const std::vector<double> &X_inp,
                                                 const std::size_t n_samples) const
{
    // Check fitted
    if (!is_fitted_)
    {
        throw std::logic_error("Model is not fitted. Call fit() before predict()");
    }

    const std::size_t total = n_samples * n_features_;
    std::vector<scalar_t> X(total);
    for (std::size_t i = 0; i < total; ++i)
    {
        X[i] = static_cast<scalar_t>(X_inp[i]);
    }

    // Check shapes
    check_shapes_predict(X, n_samples, n_features_);

    std::vector<scalar_t> preds(n_samples, scalar_t{0});
    // predictions: p = Xw + b
    calculate_predictions(X, n_samples, n_features_, preds);

    // Convert to double
    std::vector<double> preds_out(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i)
    {
        preds_out[i] = static_cast<double>(preds[i]);
    }

    return preds_out;
}

void LinearRegressionV22::gradient_descent_step(
    const std::vector<scalar_t> &X,
    const std::vector<scalar_t> &y,
    std::size_t n_samples,
    std::size_t n_features,
    std::vector<scalar_t> &preds,
    std::vector<scalar_t> &errors,
    std::vector<scalar_t> &grads)
{
    const scalar_t *X_ptr = X.data();
    const scalar_t *y_ptr = y.data();
    scalar_t *w_ptr = weights_.data();
    scalar_t *p_ptr = preds.data();
    scalar_t *e_ptr = errors.data();
    scalar_t *g_ptr = grads.data();
    scalar_t bias_gradient = scalar_t{0};

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
    std::fill(grads.begin(), grads.end(), scalar_t{0});

#pragma omp parallel
    {
        std::vector<scalar_t> g_loc(n_features, scalar_t{0});
        scalar_t *g_loc_ptr = g_loc.data();

#pragma omp for nowait schedule(static)
        for (std::size_t i = 0; i < n_samples; i++)
        {
            const scalar_t e_i = e_ptr[i];
            const scalar_t *x_row = X_ptr + i * n_features;
            for (std::size_t j = 0; j < n_features; j++)
            {
                g_loc_ptr[j] += x_row[j] * e_i;
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
    const scalar_t inv_n = scalar_t(1) / static_cast<scalar_t>(n_samples);
    for (std::size_t j = 0; j < n_features; j++)
    {
        g_ptr[j] *= inv_n;
    }

    // bias
    for (std::size_t i = 0; i < n_samples; i++)
    {
        bias_gradient += e_ptr[i];
    }
    bias_gradient *= inv_n;

    // update weights
#pragma omp parallel for
    for (std::size_t j = 0; j < n_features; j++)
    {
        w_ptr[j] -= learning_rate_ * g_ptr[j];
    }

    // update bias
    bias_ -= learning_rate_ * bias_gradient;
}

void LinearRegressionV22::calculate_predictions(const std::vector<scalar_t> &X,
                                                std::size_t n_samples,
                                                std::size_t n_features,
                                                std::vector<scalar_t> &preds) const
{
    const scalar_t *X_ptr = X.data();
    const scalar_t *w_ptr = weights_.data();
    scalar_t *p_ptr = preds.data();
#pragma omp parallel for
    for (std::size_t i = 0; i < n_samples; i++)
    {
        const scalar_t *row_ptr = X_ptr + i * n_features;
        p_ptr[i] = dot_row(row_ptr, w_ptr, n_features) + bias_;
    }
}

LinearRegressionV22::scalar_t LinearRegressionV22::dot_row(const scalar_t *a_ptr,
                                                           const scalar_t *b_ptr,
                                                           std::size_t n_features) const
{
    LinearRegressionV22::scalar_t dotProduct{0};

    for (std::size_t i = 0; i < n_features; i++)
    {
        dotProduct += a_ptr[i] * b_ptr[i];
    }

    return dotProduct;
}

void LinearRegressionV22::check_shapes_fit(const std::vector<scalar_t> &X,
                                           const std::vector<scalar_t> &y,
                                           std::size_t n_samples,
                                           std::size_t n_features) const
{
    if (n_samples == 0)
    {
        throw std::invalid_argument("n_samples must be greater than zero");
    }
    if (n_features == 0)
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

void LinearRegressionV22::check_shapes_predict(const std::vector<scalar_t> &X,
                                               std::size_t n_samples,
                                               std::size_t n_features) const
{
    if (n_samples == 0)
    {
        throw std::invalid_argument("n_samples must be greater than zero");
    }
    if (n_features == 0)
    {
        throw std::invalid_argument("n_features must be greater than zero");
    }
    if (X.size() != n_samples * n_features)
    {
        throw std::invalid_argument("X size does not match n_samples * n_features");
    }
}
