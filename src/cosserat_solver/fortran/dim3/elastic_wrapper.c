#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Fortran function declarations
void greens_displacement_force_wrapper(double x[3], double omega, double rho, double lam, double mu,
                          double G_real[3][3], double G_imag[3][3]);

// Python wrapper for greens_displacement_force
static PyObject* py_greens_displacement_force(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    double omega, rho, lam, mu;
    double x[3];
    double G_real[3][3], G_imag[3][3];

    if (!PyArg_ParseTuple(args, "Odddd", &x_obj, &omega, &rho, &lam, &mu)) {
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

    greens_displacement_force_wrapper(x, omega, rho, lam, mu, G_real, G_imag);

    // Convert 3x3 Fortran array to nested Python tuple
    PyObject* rows = PyTuple_New(3);
    for (int i = 0; i < 3; ++i) {
        PyObject* row = PyTuple_New(3);
        for (int j = 0; j < 3; ++j) {
            PyTuple_SET_ITEM(row, j, PyComplex_FromDoubles(G_real[i][j], G_imag[i][j]));
        }
        PyTuple_SET_ITEM(rows, i, row);
    }
    return rows;
}


static PyMethodDef ElasticMethods[] = {
    {"greens_displacement_force", py_greens_displacement_force, METH_VARARGS,
     "Compute Green's function for displacement force"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef elasticmodule = {
    PyModuleDef_HEAD_INIT,
    "elastic_core_3d",
    "Elastic Green's functions",
    -1,
    ElasticMethods
};

PyMODINIT_FUNC PyInit_elastic_core_3d(void) {
    return PyModule_Create(&elasticmodule);
}
