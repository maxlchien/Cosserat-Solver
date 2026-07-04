#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Fortran function declarations
void greens_displacement_force_wrapper(double x[3], double omega, double rho, double lam, double mu,
                          double G_real[3][3], double G_imag[3][3]);
void greens_displacement_force_vectorized_wrapper(double x[3], double omega[], int n_omega,
                          double rho, double lam, double mu,
                          int force_use_openmp, int force_no_openmp,
                          double G_real[/* n_omega * 3 * 3 */], double G_imag[/* n_omega * 3 * 3 */]);

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

// Python wrapper for vectorized greens_displacement_force
static PyObject* py_greens_displacement_force_vectorized(PyObject* self, PyObject* args) {
    PyObject* x_obj;
    PyObject* omega_obj;
    double rho, lam, mu;
    int force_use_openmp = 0;  /* Default: auto-decide */
    int force_no_openmp = 0;   /* Default: auto-decide */
    double x[3];

    if (!PyArg_ParseTuple(args, "OOdddii", &x_obj, &omega_obj, &rho, &lam, &mu, &force_use_openmp, &force_no_openmp)) {
        return NULL;
    }

    // Extract x values
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

    /* Allocate arrays */
    double* omega = (double*)malloc(n_omega * sizeof(double));
    double* G_real = (double*)malloc(n_omega * 3 * 3 * sizeof(double));
    double* G_imag = (double*)malloc(n_omega * 3 * 3 * sizeof(double));

    if (!omega || !G_real || !G_imag) {
        free(omega);
        free(G_real);
        free(G_imag);
        return PyErr_NoMemory();
    }

    /* Extract omega values */
    for (Py_ssize_t i = 0; i < n_omega; ++i) {
        PyObject* omega_item = PySequence_GetItem(omega_obj, i);
        if (!PyFloat_Check(omega_item)) {
            PyErr_SetString(PyExc_ValueError, "omega elements must be float");
            Py_DECREF(omega_item);
            free(omega);
            free(G_real);
            free(G_imag);
            return NULL;
        }
        omega[i] = PyFloat_AsDouble(omega_item);
        Py_DECREF(omega_item);
    }

    /* Call Fortran routine */
    greens_displacement_force_vectorized_wrapper(x, omega, (int32_t)n_omega, rho, lam, mu,
                                           force_use_openmp, force_no_openmp, G_real, G_imag);

    // Convert Nx3x3 Fortran array to nested Python tuple
    PyObject* result = PyTuple_New(n_omega);
    for (Py_ssize_t i = 0; i < n_omega; ++i) {
        PyObject* matrix = PyTuple_New(3);
        for (int row = 0; row < 3; ++row) {
            PyObject* row_tuple = PyTuple_New(3);
            for (int col = 0; col < 3; ++col) {
                /* Access G_real and G_imag in Fortran column-major order */
                double real_val = G_real[i + row * n_omega + col * n_omega * 3];
                double imag_val = G_imag[i + row * n_omega + col * n_omega * 3];
                PyTuple_SET_ITEM(row_tuple, col, PyComplex_FromDoubles(real_val, imag_val));
            }
            PyTuple_SET_ITEM(matrix, row, row_tuple);
        }
        PyTuple_SET_ITEM(result, i, matrix);
    }

    /* Clean up */
    free(omega);
    free(G_real);
    free(G_imag);

    return result;
}

static PyMethodDef ElasticMethods[] = {
    {"greens_displacement_force", py_greens_displacement_force, METH_VARARGS,
     "Compute Green's function for displacement force"},
    {"greens_displacement_force_vectorized", py_greens_displacement_force_vectorized, METH_VARARGS,
     "Compute Green's function for displacement force (vectorized)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef elasticmodule = {
    PyModuleDef_HEAD_INIT,
    "elastic_core",
    "Elastic Green's functions",
    -1,
    ElasticMethods
};

PyMODINIT_FUNC PyInit_elastic_core(void) {
    return PyModule_Create(&elasticmodule);
}
