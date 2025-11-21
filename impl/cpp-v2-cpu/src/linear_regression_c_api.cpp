#include "linear_regression_c_api.hpp"
#include "linear_regression_v21.hpp"
#include "linear_regression_v22.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

namespace
{
    thread_local std::string g_last_error;

    void clear_error() { g_last_error.clear(); }

    void set_error(const char *message)
    {
        g_last_error = message ? message : "Unknown error";
    }

    LinearRegressionV21 *as_model_v21(LinearRegressionV21Handle handle)
    {
        return reinterpret_cast<LinearRegressionV21 *>(handle);
    }

    LinearRegressionV22 *as_model_v22(LinearRegressionV22Handle handle)
    {
        return reinterpret_cast<LinearRegressionV22 *>(handle);
    }

    int handle_exception(const std::exception &ex)
    {
        set_error(ex.what());
        return -1;
    }

    int handle_unknown_exception()
    {
        set_error("Unknown exception");
        return -1;
    }
}

extern "C"
{

    // --- V21 ---

    LinearRegressionV21Handle lr_create(std::size_t iterations, double learning_rate)
    {
        try
        {
            auto *model = new LinearRegressionV21(iterations, learning_rate);
            clear_error();
            return reinterpret_cast<LinearRegressionV21Handle>(model);
        }
        catch (const std::exception &ex)
        {
            set_error(ex.what());
        }
        catch (...)
        {
            set_error("Unknown exception");
        }
        return nullptr;
    }

    void lr_destroy(LinearRegressionV21Handle handle)
    {
        if (!handle)
            return;
        auto *model = as_model_v21(handle);
        delete model;
    }

    int lr_fit(
        LinearRegressionV21Handle handle,
        const double *X,
        const double *y,
        std::size_t n_samples,
        std::size_t n_features)
    {
        if (!handle)
        {
            set_error("LinearRegressionV21 handle is null");
            return -1;
        }
        if (!X || !y)
        {
            set_error("Input pointers cannot be null");
            return -1;
        }
        try
        {
            auto *model = as_model_v21(handle);
            const std::size_t total = n_samples * n_features;
            std::vector<double> x_vec(X, X + total);
            std::vector<double> y_vec(y, y + n_samples);
            model->fit(x_vec, y_vec, n_samples, n_features);
            clear_error();
            return 0;
        }
        catch (const std::exception &ex)
        {
            return handle_exception(ex);
        }
        catch (...)
        {
            return handle_unknown_exception();
        }
    }

    int lr_predict(
        LinearRegressionV21Handle handle,
        const double *X,
        std::size_t n_samples,
        std::size_t n_features,
        double *predictions)
    {
        if (!handle)
        {
            set_error("LinearRegressionV21 handle is null");
            return -1;
        }
        if (!X || !predictions)
        {
            set_error("Input pointers cannot be null");
            return -1;
        }
        try
        {
            auto *model = as_model_v21(handle);
            const std::size_t total = n_samples * n_features;
            std::vector<double> x_vec(X, X + total);
            auto preds = model->predict(x_vec, n_samples);
            std::copy(preds.begin(), preds.end(), predictions);
            clear_error();
            return 0;
        }
        catch (const std::exception &ex)
        {
            return handle_exception(ex);
        }
        catch (...)
        {
            return handle_unknown_exception();
        }
    }

    // --- V22 ---

    LinearRegressionV22Handle lr_v22_create(std::size_t iterations, double learning_rate)
    {
        try
        {
            auto *model = new LinearRegressionV22(iterations, learning_rate);
            clear_error();
            return reinterpret_cast<LinearRegressionV22Handle>(model);
        }
        catch (const std::exception &ex)
        {
            set_error(ex.what());
        }
        catch (...)
        {
            set_error("Unknown exception");
        }
        return nullptr;
    }

    void lr_v22_destroy(LinearRegressionV22Handle handle)
    {
        if (!handle)
            return;
        auto *model = as_model_v22(handle);
        delete model;
    }

    int lr_v22_fit(
        LinearRegressionV22Handle handle,
        const double *X,
        const double *y,
        std::size_t n_samples,
        std::size_t n_features)
    {
        if (!handle)
        {
            set_error("LinearRegressionV22 handle is null");
            return -1;
        }
        if (!X || !y)
        {
            set_error("Input pointers cannot be null");
            return -1;
        }
        try
        {
            auto *model = as_model_v22(handle);
            const std::size_t total = n_samples * n_features;
            std::vector<double> x_vec(X, X + total);
            std::vector<double> y_vec(y, y + n_samples);
            model->fit(x_vec, y_vec, n_samples, n_features);
            clear_error();
            return 0;
        }
        catch (const std::exception &ex)
        {
            return handle_exception(ex);
        }
        catch (...)
        {
            return handle_unknown_exception();
        }
    }

    int lr_v22_predict(
        LinearRegressionV22Handle handle,
        const double *X,
        std::size_t n_samples,
        std::size_t n_features,
        double *predictions)
    {
        if (!handle)
        {
            set_error("LinearRegressionV22 handle is null");
            return -1;
        }
        if (!X || !predictions)
        {
            set_error("Input pointers cannot be null");
            return -1;
        }
        try
        {
            auto *model = as_model_v22(handle);
            const std::size_t total = n_samples * n_features;
            std::vector<double> x_vec(X, X + total);
            auto preds = model->predict(x_vec, n_samples);
            std::copy(preds.begin(), preds.end(), predictions);
            clear_error();
            return 0;
        }
        catch (const std::exception &ex)
        {
            return handle_exception(ex);
        }
        catch (...)
        {
            return handle_unknown_exception();
        }
    }

    const char *lr_last_error()
    {
        return g_last_error.empty() ? nullptr : g_last_error.c_str();
    }

} // extern "C"
