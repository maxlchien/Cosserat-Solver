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

/* ============================================================
   Module definition
   ============================================================ */

static PyMethodDef IntegratorMethods[] = {
    {"denom", py_denom, METH_VARARGS,
     "Evaluate denominator"},
    {"denom_prime", py_denom_prime, METH_VARARGS,
     "Evaluate derivative of denominator"},

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
