#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *LinearRegressionV21Handle;

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

    const char *lr_last_error();

#ifdef __cplusplus
}
#endif
