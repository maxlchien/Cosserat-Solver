#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Fortran function declarations
void c1_squared_wrapper(double rho, double lam, double mu, double* result);
void c2_squared_wrapper(double rho, double mu, double nu, double* result);
void c3_squared_wrapper(double J, double lam_c, double mu_c, double* result);
void c4_squared_wrapper(double J, double mu_c, double nu_c, double* result);
void w0_squared_wrapper(double nu, double J, double* result);
void dispersion_r_wrapper(double omega, double rho, double lam, double mu, double nu,
                         double J, double lam_c, double mu_c, double nu_c,
                         double* result);
void dispersion_s_wrapper(double omega, double rho, double lam, double mu, double nu,
                         double J, double lam_c, double mu_c, double nu_c,
                         double* result);
void k1_squared_wrapper(double omega, double rho, double lam, double mu,
                        double* result_real, double* result_imag);
void k2_squared_wrapper(double omega, double rho, double lam, double mu, double nu,
                        double J, double lam_c, double mu_c, double nu_c,
                        double* result_real, double* result_imag);
void k3_squared_wrapper(double omega, double nu, double J, double lam_c, double mu_c,
                        double* result_real, double* result_imag);
void k4_squared_wrapper(double omega, double rho, double lam, double mu, double nu,
                        double J, double lam_c, double mu_c, double nu_c,
                        double* result_real, double* result_imag);
void k1_wrapper(double omega, double rho, double lam, double mu,
                double* result_real, double* result_imag);
void k2_wrapper(double omega, double rho, double lam, double mu, double nu,
                double J, double lam_c, double mu_c, double nu_c,
                double* result_real, double* result_imag);
void k3_wrapper(double omega, double nu, double J, double lam_c, double mu_c,
                double* result_real, double* result_imag);
void k4_wrapper(double omega, double rho, double lam, double mu, double nu,
                double J, double lam_c, double mu_c, double nu_c,
                double* result_real, double* result_imag);

// Python wrapper for c1_squared
static PyObject* py_c1_squared(PyObject* self, PyObject* args) {
    double rho, lam, mu;
    double result;

    if (!PyArg_ParseTuple(args, "ddd", &rho, &lam, &mu)) {
        return NULL;
    }

    c1_squared_wrapper(rho, lam, mu, &result);

    return Py_BuildValue("d", result);
}

// Python wrapper for c2_squared
static PyObject* py_c2_squared(PyObject* self, PyObject* args) {
    double rho, mu, nu;
    double result;

    if (!PyArg_ParseTuple(args, "ddd", &rho, &mu, &nu)) {
        return NULL;
    }

    c2_squared_wrapper(rho, mu, nu, &result);

    return Py_BuildValue("d", result);
}

// Python wrapper for c3_squared
static PyObject* py_c3_squared(PyObject* self, PyObject* args) {
    double J, lam_c, mu_c;
    double result;

    if (!PyArg_ParseTuple(args, "ddd", &J, &lam_c, &mu_c)) {
        return NULL;
    }

    c3_squared_wrapper(J, lam_c, mu_c, &result);

    return Py_BuildValue("d", result);
}

// Python wrapper for c4_squared
static PyObject* py_c4_squared(PyObject* self, PyObject* args) {
    double J, mu_c, nu_c;
    double result;

    if (!PyArg_ParseTuple(args, "ddd", &J, &mu_c, &nu_c)) {
        return NULL;
    }

    c4_squared_wrapper(J, mu_c, nu_c, &result);

    return Py_BuildValue("d", result);
}

// Python wrapper for w0_squared
static PyObject* py_w0_squared(PyObject* self, PyObject* args) {
    double nu, J;
    double result;

    if (!PyArg_ParseTuple(args, "dd", &nu, &J)) {
        return NULL;
    }

    w0_squared_wrapper(nu, J, &result);

    return Py_BuildValue("d", result);
}

// Python wrapper for dispersion_r
static PyObject* py_dispersion_r(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double result;

    if (!PyArg_ParseTuple(args, "ddddddddd", &omega, &rho, &lam, &mu, &nu,
                          &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    dispersion_r_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &result);

    return Py_BuildValue("d", result);
}

// Python wrapper for dispersion_s
static PyObject* py_dispersion_s(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double result;

    if (!PyArg_ParseTuple(args, "ddddddddd", &omega, &rho, &lam, &mu, &nu,
                          &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    dispersion_s_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, &result);

    return Py_BuildValue("d", result);
}

// Python wrapper for k1_squared
static PyObject* py_k1_squared(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "dddd", &omega, &rho, &lam, &mu)) {
        return NULL;
    }

    k1_squared_wrapper(omega, rho, lam, mu, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for k2_squared
static PyObject* py_k2_squared(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddddddd", &omega, &rho, &lam, &mu, &nu,
                          &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    k2_squared_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                       &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for k3_squared
static PyObject* py_k3_squared(PyObject* self, PyObject* args) {
    double omega, nu, J, lam_c, mu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddd", &omega, &nu, &J, &lam_c, &mu_c)) {
        return NULL;
    }

    k3_squared_wrapper(omega, nu, J, lam_c, mu_c, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for k4_squared
static PyObject* py_k4_squared(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddddddd", &omega, &rho, &lam, &mu, &nu,
                          &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    k4_squared_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                       &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for k1
static PyObject* py_k1(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "dddd", &omega, &rho, &lam, &mu)) {
        return NULL;
    }

    k1_wrapper(omega, rho, lam, mu, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for k2
static PyObject* py_k2(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddddddd", &omega, &rho, &lam, &mu, &nu,
                          &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    k2_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
               &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for k3
static PyObject* py_k3(PyObject* self, PyObject* args) {
    double omega, nu, J, lam_c, mu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddd", &omega, &nu, &J, &lam_c, &mu_c)) {
        return NULL;
    }

    k3_wrapper(omega, nu, J, lam_c, mu_c, &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}

// Python wrapper for k4
static PyObject* py_k4(PyObject* self, PyObject* args) {
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double result_real, result_imag;

    if (!PyArg_ParseTuple(args, "ddddddddd", &omega, &rho, &lam, &mu, &nu,
                          &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    k4_wrapper(omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
               &result_real, &result_imag);

    return Py_BuildValue("(dd)", result_real, result_imag);
}


static PyMethodDef DispersionMethods[] = {
    {"c1_squared", py_c1_squared, METH_VARARGS,
     "Compute c1_squared dispersion relation"},
    {"c2_squared", py_c2_squared, METH_VARARGS,
     "Compute c2_squared dispersion relation"},
    {"c3_squared", py_c3_squared, METH_VARARGS,
     "Compute c3_squared dispersion relation"},
    {"c4_squared", py_c4_squared, METH_VARARGS,
     "Compute c4_squared dispersion relation"},
    {"w0_squared", py_w0_squared, METH_VARARGS,
     "Compute cutoff frequency w_0^2"},
    {"dispersion_r", py_dispersion_r, METH_VARARGS,
     "Compute dispersion relation coefficient r according to Eringen (1999)"},
    {"dispersion_s", py_dispersion_s, METH_VARARGS,
     "Compute dispersion relation coefficient s according to Eringen (1999)"},
    {"k1_squared", py_k1_squared, METH_VARARGS,
     "Compute k1_squared dispersion relation"},
    {"k2_squared", py_k2_squared, METH_VARARGS,
     "Compute k2_squared dispersion relation"},
    {"k3_squared", py_k3_squared, METH_VARARGS,
     "Compute k3_squared dispersion relation"},
    {"k4_squared", py_k4_squared, METH_VARARGS,
     "Compute k4_squared dispersion relation"},
    {"k1", py_k1, METH_VARARGS,
     "Compute k1 dispersion relation"},
    {"k2", py_k2, METH_VARARGS,
     "Compute k2 dispersion relation"},
    {"k3", py_k3, METH_VARARGS,
     "Compute k3 dispersion relation"},
    {"k4", py_k4, METH_VARARGS,
     "Compute k4 dispersion relation"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dispersionmodule = {
    PyModuleDef_HEAD_INIT,
    "dispersion_core",
    "Cosserat dispersion relation",
    -1,
    DispersionMethods
};

PyMODINIT_FUNC PyInit_dispersion_core(void) {
    return PyModule_Create(&dispersionmodule);
}
