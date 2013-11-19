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
#include <libsvgige/svgige.h>

#include "svs_core.h"

#define MAX_NETWORK_ADAPTERS    5
#define DISCOVERY_TIMEOUT       1000    /* Wait 1s for cameras */

/*
 * Callback for discoverCameras()
 *
 * A PyList is passed in as the context, upon which cameras are appended.
 *
 * This callback is only called from within discoverCameras(), so it does
 * not need to be thread-safe or reentrant (XXX: I think)
 */
SVGigE_RETURN camera_discovery_callback(SVGigE_SIGNAL *signal, void *context) {
    PyObject *dict;
    PyObject *list = context;
    SVGigE_CAMERA *camera;
    char buf[16];

    if (signal->SignalType != SVGigE_SIGNAL_CAMERA_FOUND) {
        /* Set an exception, though it may not be noticed */
        PyErr_Format(SVSError, "Wrong signal type received in %s: %d",
                     __func__, signal->SignalType);
        return SVGigE_ERROR;
    }

    camera = signal->Data;
    dict = PyDict_New();

    ip_int_to_string(camera->localIP, buf);

    PyObject *local_ip = Py_BuildValue("s", buf);
    PyObject *ip = Py_BuildValue("s", camera->ip);
    PyObject *subnet = Py_BuildValue("s", camera->subnet);
    PyObject *mac = Py_BuildValue("s", camera->mac);
    PyObject *manufacturer = Py_BuildValue("s", camera->manufacturer);
    PyObject *model = Py_BuildValue("s", camera->model);
    PyObject *specific_information = Py_BuildValue("s", camera->specificInformation);
    PyObject *device_version = Py_BuildValue("s", camera->deviceVersion);
    PyObject *serial_number = Py_BuildValue("s", camera->serialNumber);
    PyObject *user_name = Py_BuildValue("s", camera->userName);
    PyObject *pixel_type = Py_BuildValue("i", camera->PixelType);
    PyObject *pixel_depth = Py_BuildValue("i", camera->PixelDepth);

    PyDict_SetItemString(dict, "local_ip", local_ip);
    PyDict_SetItemString(dict, "ip", ip);
    PyDict_SetItemString(dict, "subnet", subnet);
    PyDict_SetItemString(dict, "mac", mac);
    PyDict_SetItemString(dict, "manufacturer", manufacturer);
    PyDict_SetItemString(dict, "model", model);
    PyDict_SetItemString(dict, "specific_information", specific_information);
    PyDict_SetItemString(dict, "device_version", device_version);
    PyDict_SetItemString(dict, "serial_number", serial_number);
    PyDict_SetItemString(dict, "user_name", user_name);
    PyDict_SetItemString(dict, "pixel_type", pixel_type);
    PyDict_SetItemString(dict, "pixel_depth", pixel_depth);

    Py_DECREF(local_ip);
    Py_DECREF(ip);
    Py_DECREF(subnet);
    Py_DECREF(mac);
    Py_DECREF(manufacturer);
    Py_DECREF(model);
    Py_DECREF(specific_information);
    Py_DECREF(device_version);
    Py_DECREF(serial_number);
    Py_DECREF(user_name);
    Py_DECREF(pixel_type);
    Py_DECREF(pixel_depth);

    PyList_Append(list, dict);

    Py_DECREF(dict);

    return SVGigE_SUCCESS;
}

static PyObject *svs_core_camera_list(PyObject *self, PyObject *args) {
    unsigned int adapters[MAX_NETWORK_ADAPTERS] = {0};
    PyObject *list;
    int ret;

    ret = findNetworkAdapters(adapters, MAX_NETWORK_ADAPTERS);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    list = PyList_New(0);

    for (int i = 0; i < MAX_NETWORK_ADAPTERS && adapters[i]; i++) {
        ret = discoverCameras(adapters[i], DISCOVERY_TIMEOUT,
                              camera_discovery_callback, list);
        switch (ret) {
        case SVGigE_SUCCESS:
        case SVGigE_TL_CAMERA_COMMUNICATION_TIMEOUT:    /* No cameras */
            break;
        default:
            /* Don't set an exception if one was set in the callback */
            if (!PyErr_Occurred()) {
                raise_general_error(ret);
            }
            Py_DECREF(list);
            return NULL;
        }
    }

    return list;
}

PyMethodDef svs_coreMethods[] = {
    {"camera_list", svs_core_camera_list, METH_VARARGS,
        "camera_list() -> list of cameras available\n\n"
        "Gets information on all available cameras, including camera handle,\n"
        "which can be used to select a camera to open.\n\n"
        "Returns:\n"
        "    List of dictionaries with information for each available camera.\n\n"
        "Raises:\n"
        "    SVSError: An unknown error occured in the uEye SDK."
    },
    {NULL, NULL, 0, NULL}
};
