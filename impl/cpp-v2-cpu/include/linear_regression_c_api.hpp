#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *LinearRegressionHandle;

    LinearRegressionHandle lr_create(std::size_t iterations, double learning_rate);
    void lr_destroy(LinearRegressionHandle handle);

    int lr_fit(
        LinearRegressionHandle handle,
        const double *X,
        const double *y,
        std::size_t n_samples,
        std::size_t n_features);

    int lr_predict(
        LinearRegressionHandle handle,
        const double *X,
        std::size_t n_samples,
        std::size_t n_features,
        double *predictions);

    const char *lr_last_error();

#ifdef __cplusplus
}
#endif
