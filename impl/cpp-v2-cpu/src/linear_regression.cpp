#include "linear_regression.hpp"

LinearRegression::LinearRegression(std::size_t iterations, double learning_rate) : iterations_(iterations),
                                                                                   learning_rate_(learning_rate),
                                                                                   n_features_(0),
                                                                                   weights_(),
                                                                                   bias_(0.0)
{
}

void LinearRegression::fit(const std::vector<double> &X, const std::vector<double> &y, std::size_t n_samples, std::size_t features)
{
    LinearRegression::n_features_ = features;
    LinearRegression::weights_.assign(n_features_, 0.0);
    LinearRegression::bias_ = 0.0;
}

std::vector<double> LinearRegression::predict(const std::vector<double> &X, const std::size_t n_samples, const std::size_t n_features)
{
    return std::vector<double>();
}

void LinearRegression::gradient_descent_step(
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
    const double *w_ptr = weights_.data();
    double *p_ptr = preds.data();
    double *e_ptr = errors.data();
    double *g_ptr = grads.data();

// predictions: p = Xw + b
#pragma omp parallel for
    for (std::size_t i = 0; i < n_samples; i++)
    {
        const double *row_ptr = X_ptr + i * n_features;
        p_ptr[i] = dotRow(row_ptr, w_ptr, n_features) + bias_;
    }

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
            for (std::size_t i = 0; i < n_features; i++)
            {
                g_ptr[i] += g_loc_ptr[i];
            }
        }
    }
    for (std::size_t i = 0; i < n_features; i++)
    {
        g_ptr[i] = g_ptr[i] / n_samples;
    }
}

double LinearRegression::dotRow(const double *a_ptr, const double *b_ptr, std::size_t n) const
{
    double dotProduct(0.0);

    for (std::size_t i = 0; i < n; i++)
    {
        dotProduct += a_ptr[i] * b_ptr[i];
    }

    return dotProduct;
}
