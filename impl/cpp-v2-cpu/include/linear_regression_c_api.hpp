#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *LinearRegressionV21Handle;
    typedef void *LinearRegressionV22Handle;

    // V21 API
    LinearRegressionV21Handle lr_create(std::size_t iterations, double learning_rate);
    void lr_destroy(LinearRegressionV21Handle handle);

    int lr_fit(
        LinearRegressionV21Handle handle,
        const double *X,
        const double *y,
        std::size_t n_samples,
        std::size_t n_features);

    int lr_predict(
        LinearRegressionV21Handle handle,
        const double *X,
        std::size_t n_samples,
        std::size_t n_features,
        double *predictions);

    // Shared error accessor
    const char *lr_last_error();

    // V22 API
    LinearRegressionV22Handle lr_v22_create(std::size_t iterations, double learning_rate);
    void lr_v22_destroy(LinearRegressionV22Handle handle);

    int lr_v22_fit(
        LinearRegressionV22Handle handle,
        const double *X,
        const double *y,
        std::size_t n_samples,
        std::size_t n_features);

    int lr_v22_predict(
        LinearRegressionV22Handle handle,
        const double *X,
        std::size_t n_samples,
        std::size_t n_features,
        double *predictions);

#ifdef __cplusplus
}
#endif
