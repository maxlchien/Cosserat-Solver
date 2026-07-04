#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Fortran function declarations
void c_pm_wrapper(double r_real, double r_imag, int branch,
                  double rho, double mu, double nu, double J, double mu_c, double nu_c,
                  double* result_real, double* result_imag);
void c_pm_prime_wrapper(double r_real, double r_imag, int branch,
                        double rho, double mu, double nu, double J, double mu_c, double nu_c,
                        double* result_real, double* result_imag);
void dispersion_A_wrapper(double r_real, double r_imag,
                          double rho, double nu, double J,
                          double* result_real, double* result_imag);
void dispersion_B_wrapper(double r_real, double r_imag,
                          double rho, double mu, double nu, double J, double mu_c, double nu_c,
                          double* result_real, double* result_imag);
void dispersion_C_wrapper(double r_real, double r_imag,
                          double rho, double nu, double J,
                          double* result_real, double* result_imag);
void dispersion_wrapper(double r_real, double r_imag,
                        double c_real, double c_imag,
                        double rho, double mu, double nu, double J, double mu_c, double nu_c,
                        double* result_real, double* result_imag);
void dispersion_zero_wrapper(double r_real, double r_imag,
                        int branch,
                        double rho, double mu, double nu, double J, double mu_c, double nu_c,
                        double* result_real, double* result_imag);

// Python wrapper for c_pm
static PyObject* py_c_pm(PyObject* self, PyObject* args) {
    double r_real, r_imag;
    int branch;
    double rho, mu, nu, J, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddidddddd", &r_real, &r_imag, &branch,
                          &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    c_pm_wrapper(r_real, r_imag, branch, rho, mu, nu, J, mu_c, nu_c, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for c_pm_prime
static PyObject* py_c_pm_prime(PyObject* self, PyObject* args) {
    double r_real, r_imag;
    int branch;
    double rho, mu, nu, J, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddidddddd", &r_real, &r_imag, &branch,
                          &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    c_pm_prime_wrapper(r_real, r_imag, branch, rho, mu, nu, J, mu_c, nu_c, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

static PyObject* py_dispersion_A(PyObject* self, PyObject* args) {
    double r_real, r_imag;
    double rho, nu, J;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddd", &r_real, &r_imag, &rho, &nu, &J)) {
        return NULL;
    }

    dispersion_A_wrapper(r_real, r_imag, rho, nu, J, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

static PyObject* py_dispersion_B(PyObject* self, PyObject* args) {
    double r_real, r_imag;
    double rho, mu, nu, J, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "dddddddd", &r_real, &r_imag, &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    dispersion_B_wrapper(r_real, r_imag, rho, mu, nu, J, mu_c, nu_c, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

static PyObject* py_dispersion_C(PyObject* self, PyObject* args) {
    double r_real, r_imag;
    double rho, nu, J;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddd", &r_real, &r_imag, &rho, &nu, &J)) {
        return NULL;
    }

    dispersion_C_wrapper(r_real, r_imag, rho, nu, J, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

static PyObject* py_dispersion(PyObject* self, PyObject* args) {
    double r_real, r_imag;
    double c_real, c_imag;
    double rho, mu, nu, J, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "dddddddddd", &r_real, &r_imag, &c_real, &c_imag,
                          &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    dispersion_wrapper(r_real, r_imag, c_real, c_imag, rho, mu, nu, J, mu_c, nu_c, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

static PyObject* py_dispersion_zero(PyObject* self, PyObject* args) {
    double r_real, r_imag;
    int branch;
    double rho, mu, nu, J, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddidddddd", &r_real, &r_imag, &branch,
                          &rho, &mu, &nu, &J, &mu_c, &nu_c)) {
        return NULL;
    }

    dispersion_zero_wrapper(r_real, r_imag, branch, rho, mu, nu, J, mu_c, nu_c, &result_real, &result_imag);
    return Py_BuildValue("(dd)", result_real, result_imag);
}



static PyMethodDef DispersionMethods[] = {
    {"c_pm", py_c_pm, METH_VARARGS,
     "Compute c_pm dispersion relation"},
    {"c_pm_prime", py_c_pm_prime, METH_VARARGS,
     "Compute c_pm_prime derivative"},
    {"dispersion_A", py_dispersion_A, METH_VARARGS,
     "Compute dispersion coefficient A"},
    {"dispersion_B", py_dispersion_B, METH_VARARGS,
     "Compute dispersion coefficient B"},
    {"dispersion_C", py_dispersion_C, METH_VARARGS,
     "Compute dispersion coefficient C"},
    {"dispersion", py_dispersion, METH_VARARGS,
     "Compute dispersion relation"},
    {"dispersion_zero", py_dispersion_zero, METH_VARARGS,
     "Compute dispersion relation at c_pm"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dispersionmodule = {
    PyModuleDef_HEAD_INIT,
    "dispersion_core",
    "Cosserat dispersion relation solver",
    -1,
    DispersionMethods
};

PyMODINIT_FUNC PyInit_dispersion_core(void) {
    return PyModule_Create(&dispersionmodule);
}
