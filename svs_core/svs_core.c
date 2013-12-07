/*
 * Copyright (c) 2013, North Carolina State University Aerial Robotics Club
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the North Carolina State University Aerial Robotics Club
 *       nor the names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>
#include <structmember.h>

#define PY_ARRAY_UNIQUE_SYMBOL  svs_core_ARRAY_API
#include <numpy/arrayobject.h>

#include "svs_core.h"

/* SVS Exceptions */
PyObject *SVSError;
PyObject *SVSAsyncError;
PyObject *SVSNoImagesError;

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef svs_coremodule = {
    PyModuleDef_HEAD_INIT,
    "svs_core",    /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    svs_coreMethods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_svs_core(void) {
#else
PyMODINIT_FUNC initsvs_core(void) {
#endif
    PyObject* m;

    svs_core_CameraType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&svs_core_CameraType) < 0) {
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif
    }

    import_array();
    import_datetime();

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&svs_coremodule);
#else
    m = Py_InitModule("svs_core", svs_coreMethods);
#endif

    if (m == NULL) {
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif
    }

    Py_INCREF(&svs_core_CameraType);
    PyModule_AddObject(m, "Camera", (PyObject *) &svs_core_CameraType);

    add_constants(m);

    /* SVS Exceptions */
    SVSError = PyErr_NewExceptionWithDoc("svs_core.SVSError",
            "Base class for exceptions caused by an error with the SVS camera or libraries.",
            NULL, NULL);
    Py_INCREF(SVSError);
    PyModule_AddObject(m, "SVSError", SVSError);

    SVSAsyncError = PyErr_NewExceptionWithDoc("svs_core.SVSAsyncError",
            "An asynchronous exception occurred in a callback.",
            NULL, NULL);
    Py_INCREF(SVSAsyncError);
    PyModule_AddObject(m, "SVSAsyncError", SVSAsyncError);

    SVSNoImagesError = PyErr_NewExceptionWithDoc("svs_core.SVSNoImagesError",
            "Raised when no more images are available.", SVSError, NULL);
    Py_INCREF(SVSNoImagesError);
    PyModule_AddObject(m, "SVSNoImagesError", SVSNoImagesError);

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

int main(int argc, char *argv[]) {
#if PY_MAJOR_VERSION >= 3
    wchar_t name[128];
    mbstowcs(name, argv[0], 128);
#else
    char name[128];
    strncpy(name, argv[0], 128);
#endif

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(name);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Ensure the GIL has been created */
    if (!PyEval_ThreadsInitialized()) {
        PyEval_InitThreads();
    }

    /* Add a static module */
#if PY_MAJOR_VERSION >= 3
    PyInit_svs_core();
#else
    initsvs_core();
#endif

    return 0;
}
