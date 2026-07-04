#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <complex.h>

/* ============================================================
   Fortran function declarations (bind(C, name="..."))
   ============================================================ */

/* Denominator and derivative */
void denom(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    int32_t branch,
    double rho,
    double mu,
    double nu,
    double J,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void denom_prime(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    int32_t branch,
    double rho,
    double mu,
    double nu,
    double J,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

/* Poles and branches */
void get_r2_poles_and_branches(
    double omega_re,
    double omega_im,
    double r2_re[2],
    double r2_im[2],
    int32_t branch[2],
    int32_t *n,
    double rho,
    double mu,
    double nu,
    double J,
    double mu_c,
    double nu_c
);

void pick_pole(
    double r2_re,
    double r2_im,
    double omega_re,
    double omega_im,
    double *out_re,
    double *out_im
);

/* Integrals */
void integral_3_0(
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integral_3_2(
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integral_2_1(
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integral_1_0(
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

/* Integrand kernels */
void integrand_3_0(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integrand_3_2(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integrand_2_1(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integrand_1_0(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

/* Full integrand kernels (with denominator) */
void integrand_3_0_full(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integrand_3_2_full(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integrand_2_1_full(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void integrand_1_0_full(
    double r_re,
    double r_im,
    double omega_re,
    double omega_im,
    double normx_re,
    double normx_im,
    int32_t branch,
    double rho,
    double lam,
    double mu,
    double nu,
    double J,
    double lam_c,
    double mu_c,
    double nu_c,
    double *out_re,
    double *out_im
);

void greens_x_omega_P(
    double x[2], double omega_re, double omega_im,
    double rho, double lam, double mu, double nu, double J,
    double lam_c, double mu_c, double nu_c,
    double complex G[3*3]
);
void greens_x_omega_plus(
    double x[2], double omega_re, double omega_im,
    double rho, double lam, double mu, double nu, double J,
    double lam_c, double mu_c, double nu_c,
    double complex G[3*3]
);
void greens_x_omega_minus(
    double x[2], double omega_re, double omega_im,
    double rho, double lam, double mu, double nu, double J,
    double lam_c, double mu_c, double nu_c,
    double complex G[3*3]
);
void greens_x_omega(
    double x[2], double omega_re, double omega_im,
    double rho, double lam, double mu, double nu, double J,
    double lam_c, double mu_c, double nu_c,
    double complex G[3*3]
);
void greens_x_omega_vectorized(
    double x[2],
    double omega_re[], double omega_im[], int32_t n_omega,
    double rho, double lam, double mu, double nu, double J,
    double lam_c, double mu_c, double nu_c,
    int force_use_openmp, int force_no_openmp,
    double complex G[/* n_omega * 3 * 3 */]
);

/* ============================================================
   LowLevelCallable support for integrands
   ============================================================ */

typedef struct integrand_ctx {
    double omega_re;
    double omega_im;
    double normx_re;
    double normx_im;
    int32_t branch;
    double rho;
    double lam;
    double mu;
    double nu;
    double J;
    double lam_c;
    double mu_c;
    double nu_c;
    double contour_shift;
    int32_t integrand_id; /* 0:3_0, 1:3_2, 2:2_1, 3:1_0 */
    int32_t component;    /* 0: real, 1: imag */
} integrand_ctx;

static double integrand_llc(double x, void* user_data) {
    integrand_ctx* ctx = (integrand_ctx*)user_data;
    double val_re = 0.0, val_im = 0.0;
    double r_re = x;
    double r_im = ctx->contour_shift;

    switch (ctx->integrand_id) {
        case 0:
            integrand_3_0_full(r_re, r_im,
                          ctx->omega_re, ctx->omega_im,
                          ctx->normx_re, ctx->normx_im,
                          ctx->branch,
                          ctx->rho, ctx->lam, ctx->mu, ctx->nu, ctx->J,
                          ctx->lam_c, ctx->mu_c, ctx->nu_c,
                          &val_re, &val_im);
            break;
        case 1:
            integrand_3_2_full(r_re, r_im,
                          ctx->omega_re, ctx->omega_im,
                          ctx->normx_re, ctx->normx_im,
                          ctx->branch,
                          ctx->rho, ctx->lam, ctx->mu, ctx->nu, ctx->J,
                          ctx->lam_c, ctx->mu_c, ctx->nu_c,
                          &val_re, &val_im);
            break;
        case 2:
            integrand_2_1_full(r_re, r_im,
                          ctx->omega_re, ctx->omega_im,
                          ctx->normx_re, ctx->normx_im,
                          ctx->branch,
                          ctx->rho, ctx->lam, ctx->mu, ctx->nu, ctx->J,
                          ctx->lam_c, ctx->mu_c, ctx->nu_c,
                          &val_re, &val_im);
            break;
        case 3:
            integrand_1_0_full(r_re, r_im,
                          ctx->omega_re, ctx->omega_im,
                          ctx->normx_re, ctx->normx_im,
                          ctx->branch,
                          ctx->rho, ctx->lam, ctx->mu, ctx->nu, ctx->J,
                          ctx->lam_c, ctx->mu_c, ctx->nu_c,
                          &val_re, &val_im);
            break;
        default:
            return 0.0;
    }

    return (ctx->component == 0) ? val_re : val_im;
}

/* ============================================================
   Python wrappers
   ============================================================ */

/* ------------------ denom ------------------ */

static PyObject* py_denom(PyObject* self, PyObject* args) {
    double r_re, r_im, omega_re, omega_im;
    double rho, mu, nu, J, mu_c, nu_c;
    int branch;
    double out_re, out_im;

    if (!PyArg_ParseTuple(args, "ddddidddddd",
                          &r_re, &r_im,
                          &omega_re, &omega_im,
                          &branch,
                          &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    denom(r_re, r_im, omega_re, omega_im, (int32_t)branch,
          rho, mu, nu, J, mu_c, nu_c,
          &out_re, &out_im);

    return Py_BuildValue("(dd)", out_re, out_im);
}

static PyObject* py_denom_prime(PyObject* self, PyObject* args) {
    double r_re, r_im, omega_re, omega_im;
    double rho, mu, nu, J, mu_c, nu_c;
    int branch;
    double out_re, out_im;

    if (!PyArg_ParseTuple(args, "ddddidddddd",
                          &r_re, &r_im,
                          &omega_re, &omega_im,
                          &branch,
                          &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    denom_prime(r_re, r_im, omega_re, omega_im, (int32_t)branch,
                rho, mu, nu, J, mu_c, nu_c,
                &out_re, &out_im);

    return Py_BuildValue("(dd)", out_re, out_im);
}

/* ------------------ integrands ------------------ */

static PyObject* py_integrand_3_0(PyObject* self, PyObject* args) {
    double r_re, r_im, omega_re, omega_im, normx_re, normx_im;
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    int branch;
    double out_re, out_im;

    if (!PyArg_ParseTuple(args, "ddddddidddddddd",
                          &r_re, &r_im,
                          &omega_re, &omega_im,
                          &normx_re, &normx_im,
                          &branch,
                          &rho, &lam, &mu, &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    integrand_3_0(r_re, r_im,
                  omega_re, omega_im,
                  normx_re, normx_im,
                  (int32_t)branch,
                  rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                  &out_re, &out_im);

    return Py_BuildValue("(dd)", out_re, out_im);
}

static PyObject* py_integrand_3_2(PyObject* self, PyObject* args) {
    double r_re, r_im, omega_re, omega_im, normx_re, normx_im;
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    int branch;
    double out_re, out_im;

    if (!PyArg_ParseTuple(args, "ddddddidddddddd",
                          &r_re, &r_im,
                          &omega_re, &omega_im,
                          &normx_re, &normx_im,
                          &branch,
                          &rho, &lam, &mu, &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    integrand_3_2(r_re, r_im,
                  omega_re, omega_im,
                  normx_re, normx_im,
                  (int32_t)branch,
                  rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                  &out_re, &out_im);

    return Py_BuildValue("(dd)", out_re, out_im);
}

static PyObject* py_integrand_2_1(PyObject* self, PyObject* args) {
    double r_re, r_im, omega_re, omega_im, normx_re, normx_im;
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    int branch;
    double out_re, out_im;

    if (!PyArg_ParseTuple(args, "ddddddidddddddd",
                          &r_re, &r_im,
                          &omega_re, &omega_im,
                          &normx_re, &normx_im,
                          &branch,
                          &rho, &lam, &mu, &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    integrand_2_1(r_re, r_im,
                  omega_re, omega_im,
                  normx_re, normx_im,
                  (int32_t)branch,
                  rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                  &out_re, &out_im);

    return Py_BuildValue("(dd)", out_re, out_im);
}

static PyObject* py_integrand_1_0(PyObject* self, PyObject* args) {
    double r_re, r_im, omega_re, omega_im, normx_re, normx_im;
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    int branch;
    double out_re, out_im;

    if (!PyArg_ParseTuple(args, "ddddddidddddddd",
                          &r_re, &r_im,
                          &omega_re, &omega_im,
                          &normx_re, &normx_im,
                          &branch,
                          &rho, &lam, &mu, &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    integrand_1_0(r_re, r_im,
                  omega_re, omega_im,
                  normx_re, normx_im,
                  (int32_t)branch,
                  rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                  &out_re, &out_im);

    return Py_BuildValue("(dd)", out_re, out_im);
}

static PyObject* py_integrand_llc_capsule(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    return PyCapsule_New((void*)integrand_llc, "double (double, void *)", NULL);
}

/* ------------------ poles ------------------ */

static PyObject* py_get_r2_poles_and_branches(PyObject* self, PyObject* args) {
    double omega_re, omega_im;
    double rho, mu, nu, J, mu_c, nu_c;
    double r2_re[2], r2_im[2];
    int32_t branch[2];
    int32_t n;

    if (!PyArg_ParseTuple(args, "dddddddd", &omega_re, &omega_im,
                          &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    get_r2_poles_and_branches(
        omega_re, omega_im,
        r2_re, r2_im, branch, &n,
        rho, mu, nu, J, mu_c, nu_c
    );

    PyObject* poles = PyList_New(n);
    PyObject* branches = PyList_New(n);

    for (int i = 0; i < n; ++i) {
        PyList_SET_ITEM(poles, i,
            Py_BuildValue("(dd)", r2_re[i], r2_im[i]));
        PyList_SET_ITEM(branches, i,
            PyLong_FromLong(branch[i]));
    }

    return Py_BuildValue("(OO)", poles, branches);
}

static PyObject* py_pick_pole(PyObject* self, PyObject* args) {
    double r2_re, r2_im, omega_re, omega_im;
    double out_re, out_im;

    if (!PyArg_ParseTuple(args, "dddd",
                          &r2_re, &r2_im,
                          &omega_re, &omega_im)) {
        return NULL;
    }

    pick_pole(r2_re, r2_im, omega_re, omega_im, &out_re, &out_im);

    return Py_BuildValue("(dd)", out_re, out_im);
}

/* ------------------ integrals ------------------ */

#define DEFINE_INTEGRAL_WRAPPER(NAME)                                      \
static PyObject* py_##NAME(PyObject* self, PyObject* args) {               \
    double omega_re, omega_im, normx_re, normx_im;                         \
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;                         \
    int branch;                                                             \
    double out_re, out_im;                                                 \
                                                                           \
    if (!PyArg_ParseTuple(args, "ddddidddddddd",                            \
                          &omega_re, &omega_im,                            \
                          &normx_re, &normx_im,                            \
                          &branch,                                         \
                          &rho, &lam, &mu, &nu, &J, &lam_c, &mu_c, &nu_c)) { \
        return NULL;                                                       \
    }                                                                      \
                                                                           \
    NAME(omega_re, omega_im, normx_re, normx_im,                           \
         (int32_t)branch, rho, lam, mu, nu, J, lam_c, mu_c, nu_c,          \
         &out_re, &out_im);                                                \
                                                                           \
    return Py_BuildValue("(dd)", out_re, out_im);                           \
}

DEFINE_INTEGRAL_WRAPPER(integral_3_0)
DEFINE_INTEGRAL_WRAPPER(integral_3_2)
DEFINE_INTEGRAL_WRAPPER(integral_2_1)
DEFINE_INTEGRAL_WRAPPER(integral_1_0)

#define DEFINE_GREENS_WRAPPER(FUNCNAME)                                    \
static PyObject* py_##FUNCNAME(PyObject* self, PyObject* args) {           \
    PyObject* x_obj;                                                       \
    double omega_re, omega_im;                                             \
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;                         \
    double x[2];                                                           \
    double complex G[3*3];                                                 \
                                                                            \
    if (!PyArg_ParseTuple(args, "Odddddddddd", &x_obj,                     \
                          &omega_re, &omega_im,                            \
                          &rho, &lam, &mu, &nu, &J,                        \
                          &lam_c, &mu_c, &nu_c)) {                         \
        return NULL;                                                       \
    }                                                                      \
                                                                           \
    /* Extract two floats from x_obj */                                    \
    if (!PySequence_Check(x_obj) || PySequence_Size(x_obj) != 2) {         \
        PyErr_SetString(PyExc_ValueError, "x must be sequence of length 2"); \
        return NULL;                                                       \
    }                                                                      \
    for (int i = 0; i < 2; ++i) {                                          \
        PyObject* item = PySequence_GetItem(x_obj, i);                     \
        if (!PyFloat_Check(item)) {                                        \
            PyErr_SetString(PyExc_ValueError, "x elements must be float"); \
            Py_DECREF(item);                                               \
            return NULL;                                                   \
        }                                                                  \
        x[i] = PyFloat_AsDouble(item);                                      \
        Py_DECREF(item);                                                   \
    }                                                                      \
                                                                           \
    FUNCNAME(x, omega_re, omega_im,                                        \
             rho, lam, mu, nu, J, lam_c, mu_c, nu_c, G);                   \
                                                                           \
    /* Convert 3x3 Fortran array to nested Python tuple */                 \
    PyObject* rows = PyTuple_New(3);                                       \
    for (int i = 0; i < 3; ++i) {                                          \
        PyObject* row = PyTuple_New(3);                                    \
        for (int j = 0; j < 3; ++j) {                                      \
            PyTuple_SET_ITEM(row, j, PyComplex_FromDoubles(                \
                creal(G[i + j*3]), cimag(G[i + j*3])));                    \
        }                                                                  \
        PyTuple_SET_ITEM(rows, i, row);                                    \
    }                                                                      \
    return rows;                                                            \
}

DEFINE_GREENS_WRAPPER(greens_x_omega_P)
DEFINE_GREENS_WRAPPER(greens_x_omega_plus)
DEFINE_GREENS_WRAPPER(greens_x_omega_minus)
DEFINE_GREENS_WRAPPER(greens_x_omega)

/* Vectorized Green's function wrapper */
static PyObject* py_greens_x_omega_vectorized(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    PyObject* omega_obj;
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    int force_use_openmp = 0;  /* Default: auto-decide */
    int force_no_openmp = 0;   /* Default: auto-decide */
    double x[2];

    if (!PyArg_ParseTuple(args, "OOddddddddii", &x_obj, &omega_obj,
                          &rho, &lam, &mu, &nu, &J,
                          &lam_c, &mu_c, &nu_c, &force_use_openmp, &force_no_openmp)) {
        return NULL;
    }

    /* Extract x */
    if (!PySequence_Check(x_obj) || PySequence_Size(x_obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "x must be sequence of length 2");
        return NULL;
    }
    for (int i = 0; i < 2; ++i) {
        PyObject* item = PySequence_GetItem(x_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "x elements must be float");
            Py_DECREF(item);
            return NULL;
        }
        x[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
    }

    /* Extract omega array */
    if (!PySequence_Check(omega_obj)) {
        PyErr_SetString(PyExc_ValueError, "omega must be a sequence");
        return NULL;
    }

    Py_ssize_t n_omega = PySequence_Size(omega_obj);
    if (n_omega <= 0) {
        PyErr_SetString(PyExc_ValueError, "omega array must have at least one element");
        return NULL;
    }

    /* Allocate arrays for real and imaginary parts */
    double* omega_re = (double*)malloc(n_omega * sizeof(double));
    double* omega_im = (double*)malloc(n_omega * sizeof(double));
    double complex* G = (double complex*)malloc(n_omega * 3 * 3 * sizeof(double complex));

    if (!omega_re || !omega_im || !G) {
        free(omega_re);
        free(omega_im);
        free(G);
        return PyErr_NoMemory();
    }

    /* Extract omega values */
    for (Py_ssize_t i = 0; i < n_omega; ++i) {
        PyObject* omega_item = PySequence_GetItem(omega_obj, i);
        if (!PyComplex_Check(omega_item)) {
            PyErr_SetString(PyExc_ValueError, "omega elements must be complex");
            Py_DECREF(omega_item);
            free(omega_re);
            free(omega_im);
            free(G);
            return NULL;
        }
        omega_re[i] = PyComplex_RealAsDouble(omega_item);
        omega_im[i] = PyComplex_ImagAsDouble(omega_item);
        Py_DECREF(omega_item);
    }

    /* Call Fortran routine */
    greens_x_omega_vectorized(x, omega_re, omega_im, (int32_t)n_omega,
                              rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                              force_use_openmp, force_no_openmp, G);

    /* Convert result to Python nested list/tuple */
    PyObject* result = PyList_New(n_omega);
    for (Py_ssize_t i = 0; i < n_omega; ++i) {
        PyObject* matrix = PyTuple_New(3);
        for (int row = 0; row < 3; ++row) {
            PyObject* row_tuple = PyTuple_New(3);
            for (int col = 0; col < 3; ++col) {
                /* Access G[i, row, col] - Fortran column-major ordering */
                double complex val = G[i * 9 + row + col * 3];
                PyTuple_SET_ITEM(row_tuple, col,
                    PyComplex_FromDoubles(creal(val), cimag(val)));
            }
            PyTuple_SET_ITEM(matrix, row, row_tuple);
        }
        PyList_SET_ITEM(result, i, matrix);
    }

    /* Clean up */
    free(omega_re);
    free(omega_im);
    free(G);

    return result;
}

/* ============================================================
   Module definition
   ============================================================ */

static PyMethodDef IntegratorMethods[] = {
    {"denom", py_denom, METH_VARARGS,
     "Evaluate denominator"},
    {"denom_prime", py_denom_prime, METH_VARARGS,
     "Evaluate derivative of denominator"},

    {"integrand_3_0", py_integrand_3_0, METH_VARARGS, "Integrand 3_0"},
    {"integrand_3_2", py_integrand_3_2, METH_VARARGS, "Integrand 3_2"},
    {"integrand_2_1", py_integrand_2_1, METH_VARARGS, "Integrand 2_1"},
    {"integrand_1_0", py_integrand_1_0, METH_VARARGS, "Integrand 1_0"},
    {"integrand_llc_capsule", py_integrand_llc_capsule, METH_NOARGS,
     "Return LowLevelCallable capsule for integrands"},

    {"get_r2_poles_and_branches", py_get_r2_poles_and_branches, METH_VARARGS,
     "Compute r^2 poles and associated branches"},
    {"pick_pole", py_pick_pole, METH_VARARGS,
     "Select physical pole"},

    {"integral_3_0", py_integral_3_0, METH_VARARGS, "Integral 3_0"},
    {"integral_3_2", py_integral_3_2, METH_VARARGS, "Integral 3_2"},
    {"integral_2_1", py_integral_2_1, METH_VARARGS, "Integral 2_1"},
    {"integral_1_0", py_integral_1_0, METH_VARARGS, "Integral 1_0"},

    {"greens_x_omega_P_c", py_greens_x_omega_P, METH_VARARGS, "Green's function P"},
    {"greens_x_omega_plus_c", py_greens_x_omega_plus, METH_VARARGS, "Green's function plus"},
    {"greens_x_omega_minus_c", py_greens_x_omega_minus, METH_VARARGS, "Green's function minus"},
    {"greens_x_omega_c", py_greens_x_omega, METH_VARARGS, "Green's function full"},
    {"greens_x_omega_vectorized_c", py_greens_x_omega_vectorized, METH_VARARGS, "Vectorized Green's function"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef integratormodule = {
    PyModuleDef_HEAD_INIT,
    "integrator_core",
    "Cosserat Green's function integrator",
    -1,
    IntegratorMethods
};

PyMODINIT_FUNC PyInit_integrator_core(void) {
    return PyModule_Create(&integratormodule);
}
