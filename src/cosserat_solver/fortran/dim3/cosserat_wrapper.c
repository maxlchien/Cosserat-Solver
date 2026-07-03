#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Fortran function declarations
void greens_mixed_force_wrapper(double x[3], double omega, double rho, double lam, double mu,
                          double nu, double J, double lam_c, double mu_c, double nu_c,
                          double G_real[6][6], double G_imag[6][6]);
void greens_displacement_force_wrapper(double x[3], double omega, double rho, double lam, double mu,
                          double nu, double J, double lam_c, double mu_c, double nu_c,
                          double G_real[6][3], double G_imag[6][3]);
void greens_rotation_force_wrapper(double x[3], double omega, double rho, double lam, double mu,
                          double nu, double J, double lam_c, double mu_c, double nu_c,
                          double G_real[6][3], double G_imag[6][3]);
void greens_displacement_force_static_wrapper(double x[3], double rho, double lam, double mu, double nu,
                                         double J, double lam_c, double mu_c, double nu_c,
                                         double G_real[6][3], double G_imag[6][3]);
void greens_rotation_force_static_wrapper(double x[3], double rho, double lam, double mu, double nu,
                                         double J, double lam_c, double mu_c, double nu_c,
                                         double G_real[6][3], double G_imag[6][3]);

// Python wrapper for greens_mixed_force
static PyObject* py_greens_mixed_force(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double x[3];
    double G_real[6*6], G_imag[6*6];

    if (!PyArg_ParseTuple(args, "Oddddddddd", &x_obj, &omega, &rho, &lam, &mu,
                          &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    /* Extract three floats from x_obj */
    if (!PySequence_Check(x_obj) || PySequence_Size(x_obj) != 3) {
        PyErr_SetString(PyExc_ValueError, "x must be sequence of length 3");
        return NULL;
    }
    for (int i = 0; i < 3; ++i) {
        PyObject* item = PySequence_GetItem(x_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "x elements must be float");
            Py_DECREF(item);
            return NULL;
        }
        x[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
    }

    greens_mixed_force_wrapper(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, G_real, G_imag);

    // Convert 6x6 Fortran array to nested Python tuple
    PyObject* rows = PyTuple_New(6);
    for (int i = 0; i < 6; ++i) {
        PyObject* row = PyTuple_New(6);
        for (int j = 0; j < 6; ++j) {
            PyTuple_SET_ITEM(row, j, PyComplex_FromDoubles(G_real[i + 6 * j], G_imag[i + 6 * j]));
        }
        PyTuple_SET_ITEM(rows, i, row);
    }
    return rows;
}

// Python wrapper for greens_displacement_force
static PyObject* py_greens_displacement_force(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double x[3];
    double G_real[6*3], G_imag[6*3];

    if (!PyArg_ParseTuple(args, "Oddddddddd", &x_obj, &omega, &rho, &lam, &mu,
                          &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    /* Extract three floats from x_obj */
    if (!PySequence_Check(x_obj) || PySequence_Size(x_obj) != 3) {
        PyErr_SetString(PyExc_ValueError, "x must be sequence of length 3");
        return NULL;
    }
    for (int i = 0; i < 3; ++i) {
        PyObject* item = PySequence_GetItem(x_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "x elements must be float");
            Py_DECREF(item);
            return NULL;
        }
        x[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
    }

    greens_displacement_force_wrapper(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c,
                                      G_real, G_imag);

    // Convert 6x3 Fortran array to nested Python tuple
    PyObject* rows = PyTuple_New(6);
    for (int i = 0; i < 6; ++i) {
        PyObject* row = PyTuple_New(3);
        for (int j = 0; j < 3; ++j) {
            PyTuple_SET_ITEM(row, j, PyComplex_FromDoubles(G_real[i + 6 * j], G_imag[i + 6 * j]));
        }
        PyTuple_SET_ITEM(rows, i, row);
    }
    return rows;
}

// Python wrapper for greens_rotation_force
static PyObject* py_greens_rotation_force(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    double omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double x[3];
    double G_real[6*3], G_imag[6*3];

    if (!PyArg_ParseTuple(args, "Oddddddddd", &x_obj, &omega, &rho, &lam, &mu,
                          &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    /* Extract three floats from x_obj */
    if (!PySequence_Check(x_obj) || PySequence_Size(x_obj) != 3) {
        PyErr_SetString(PyExc_ValueError, "x must be sequence of length 3");
        return NULL;
    }
    for (int i = 0; i < 3; ++i) {
        PyObject* item = PySequence_GetItem(x_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "x elements must be float");
            Py_DECREF(item);
            return NULL;
        }
        x[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
    }

    greens_rotation_force_wrapper(x, omega, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, G_real, G_imag);

    // Convert 6x3 Fortran array to nested Python tuple
    PyObject* rows = PyTuple_New(6);
    for (int i = 0; i < 6; ++i) {
        PyObject* row = PyTuple_New(3);
        for (int j = 0; j < 3; ++j) {
            PyTuple_SET_ITEM(row, j, PyComplex_FromDoubles(G_real[i + 6 * j], G_imag[i + 6 * j]));
        }
        PyTuple_SET_ITEM(rows, i, row);
    }
    return rows;
}

// Python wrapper for greens_displacement_force_static
static PyObject* py_greens_displacement_force_static(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double x[3];
    double G_real[6*3], G_imag[6*3];

    if (!PyArg_ParseTuple(args, "Odddddddd", &x_obj, &rho, &lam, &mu, &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    /* Extract three floats from x_obj */
    if (!PySequence_Check(x_obj) || PySequence_Size(x_obj) != 3) {
        PyErr_SetString(PyExc_ValueError, "x must be sequence of length 3");
        return NULL;
    }
    for (int i = 0; i < 3; ++i) {
        PyObject* item = PySequence_GetItem(x_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "x elements must be float");
            Py_DECREF(item);
            return NULL;
        }
        x[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
    }

    greens_displacement_force_static_wrapper(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, G_real, G_imag);

    // Convert 6x3 Fortran array to nested Python tuple
    PyObject* rows = PyTuple_New(6);
    for (int i = 0; i < 6; ++i) {
        PyObject* row = PyTuple_New(3);
        for (int j = 0; j < 3; ++j) {
            PyTuple_SET_ITEM(row, j, PyComplex_FromDoubles(G_real[i + 6 * j], G_imag[i + 6 * j]));
        }
        PyTuple_SET_ITEM(rows, i, row);
    }
    return rows;
}

// Python wrapper for greens_rotation_force_static
static PyObject* py_greens_rotation_force_static(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    double rho, lam, mu, nu, J, lam_c, mu_c, nu_c;
    double x[3];
    double G_real[6*3], G_imag[6*3];

    if (!PyArg_ParseTuple(args, "Odddddddd", &x_obj, &rho, &lam, &mu, &nu, &J, &lam_c, &mu_c, &nu_c)) {
        return NULL;
    }

    /* Extract three floats from x_obj */
    if (!PySequence_Check(x_obj) || PySequence_Size(x_obj) != 3) {
        PyErr_SetString(PyExc_ValueError, "x must be sequence of length 3");
        return NULL;
    }
    for (int i = 0; i < 3; ++i) {
        PyObject* item = PySequence_GetItem(x_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "x elements must be float");
            Py_DECREF(item);
            return NULL;
        }
        x[i] = PyFloat_AsDouble(item);
        Py_DECREF(item);
    }

    greens_rotation_force_static_wrapper(x, rho, lam, mu, nu, J, lam_c, mu_c, nu_c, G_real, G_imag);

    // Convert 6x3 Fortran array to nested Python tuple
    PyObject* rows = PyTuple_New(6);
    for (int i = 0; i < 6; ++i) {
        PyObject* row = PyTuple_New(3);
        for (int j = 0; j < 3; ++j) {
            PyTuple_SET_ITEM(row, j, PyComplex_FromDoubles(G_real[i + 6 * j], G_imag[i + 6 * j]));
        }
        PyTuple_SET_ITEM(rows, i, row);
    }
    return rows;
}

static PyMethodDef CosseratMethods[] = {
    {"greens_mixed_force", py_greens_mixed_force, METH_VARARGS,
     "Compute Green's function for mixed force"},
    {"greens_displacement_force", py_greens_displacement_force, METH_VARARGS,
     "Compute Green's function for displacement force"},
    {"greens_rotation_force", py_greens_rotation_force, METH_VARARGS,
     "Compute Green's function for rotation force"},
    {"greens_displacement_force_static", py_greens_displacement_force_static, METH_VARARGS,
     "Compute static Green's function for displacement force"},
    {"greens_rotation_force_static", py_greens_rotation_force_static, METH_VARARGS,
     "Compute static Green's function for rotation force"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cosseratmodule = {
    PyModuleDef_HEAD_INIT,
    "cosserat_core_3d",
    "Cosserat Green's functions",
    -1,
    CosseratMethods
};

PyMODINIT_FUNC PyInit_cosserat_core_3d(void) {
    return PyModule_Create(&cosseratmodule);
}
