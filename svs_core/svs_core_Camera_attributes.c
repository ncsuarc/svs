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

PyObject *svs_core_Camera_getinfo(svs_core_Camera *self, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return NULL;
}

static int svs_core_Camera_setinfo(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_TypeError, "Cannot modify attribute 'info'");
    return -1;
}

PyObject *svs_core_Camera_getname(svs_core_Camera *self, void *closure) {
    Py_INCREF(self->name);
    return self->name;
}

static int svs_core_Camera_setname(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_TypeError, "Cannot modify attribute 'name'");
    return -1;
}

static PyObject *svs_core_Camera_getwidth(svs_core_Camera *self, void *closure) {
    return PyLong_FromLong(self->width);
}

static int svs_core_Camera_setwidth(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Changing image width not yet supported.");
    return -1;
}

static PyObject *svs_core_Camera_getheight(svs_core_Camera *self, void *closure) {
    return PyLong_FromLong(self->height);
}

static int svs_core_Camera_setheight(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Changing image height not yet supported.");
    return -1;
}

static PyObject *svs_core_Camera_getpixelclock(svs_core_Camera *self, void *closure) {
    int pixel_clock, ret;

    ret = Camera_getPixelClock(self->handle, &pixel_clock);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    return PyLong_FromLong(pixel_clock);
}

static int svs_core_Camera_setpixelclock(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_TypeError, "Cannot modify attribute 'pixelclock'");
    return -1;
}

static PyObject *svs_core_Camera_getcolor_mode(svs_core_Camera *self, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return NULL;
}

static int svs_core_Camera_setcolor_mode(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return -1;
}

static PyObject *svs_core_Camera_getgain(svs_core_Camera *self, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return NULL;
}

static int svs_core_Camera_setgain(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return -1;
}

static PyObject *svs_core_Camera_getexposure(svs_core_Camera *self, void *closure) {
    float exposure;
    int ret;

    /* Exposure in microseconds */
    ret = Camera_getExposureTime(self->handle, &exposure);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    /* Return exposure in milliseconds */
    return PyFloat_FromDouble(exposure/1000);
}

static int svs_core_Camera_setexposure(svs_core_Camera *self, PyObject *value, void *closure) {
    int ret;
    float exposure = PyFloat_AsDouble(value);   /* Exposure in milliseconds */
    if (PyErr_Occurred()) {
        return -1;
    }

    /* Set exposure in microseconds */
    ret = Camera_setExposureTime(self->handle, 1000*exposure);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getauto_exposure(svs_core_Camera *self, void *closure) {
    bool autoexposure;
    int ret;

    ret = Camera_getAutoGainEnabled(self->handle, &autoexposure);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    if (autoexposure) {
        Py_INCREF(Py_True);
        return Py_True;
    }

    Py_INCREF(Py_False);
    return Py_False;
}

static int svs_core_Camera_setauto_exposure(svs_core_Camera *self, PyObject *value, void *closure) {
    bool autoexposure;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure'");
        return -1;
    }

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auto exposure must be a bool.");
        return -1;
    }

    autoexposure = (value == Py_True) ? 1 : 0;

    ret = Camera_setAutoGainEnabled(self->handle, autoexposure);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getauto_exposure_brightness(svs_core_Camera *self, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return NULL;
}

static int svs_core_Camera_setauto_exposure_brightness(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return -1;
}

static PyObject *svs_core_Camera_getauto_speed(svs_core_Camera *self, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return NULL;
}

static int svs_core_Camera_setauto_speed(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return -1;
}

static PyObject *svs_core_Camera_getauto_white_balance(svs_core_Camera *self, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return NULL;
}

static int svs_core_Camera_setauto_white_balance(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return -1;
}

static PyObject *svs_core_Camera_getcolor_correction(svs_core_Camera *self, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return NULL;
}

static int svs_core_Camera_setcolor_correction(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Not yet implemented");
    return -1;
}

static PyObject *svs_core_Camera_getcontinuous_capture(svs_core_Camera *self,
                                                       void *closure) {
    int ret;
    ACQUISITION_CONTROL control;

    ret = Camera_getAcquisitionControl(self->handle, &control);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    if (control == ACQUISITION_CONTROL_START) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

static int svs_core_Camera_setcontinuous_capture(svs_core_Camera *self,
                                                 PyObject *value, void *closure) {
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'continuous_capture'");
        return -1;
    }

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Attribute 'continuous_capture' must be boolean");
        return -1;
    }

    /* Enable continuous capture */
    if (value == Py_True) {
        ret = Camera_setAcquisitionMode(self->handle,
                                        ACQUISITION_MODE_FIXED_FREQUENCY, 1);
    }
    else {
        ret = Camera_setAcquisitionControl(self->handle,
                                           ACQUISITION_CONTROL_STOP);
    }

    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

PyGetSetDef svs_core_Camera_getseters[] = {
    {"info", (getter) svs_core_Camera_getinfo, (setter) svs_core_Camera_setinfo, "Camera info", NULL},
    {"name", (getter) svs_core_Camera_getname, (setter) svs_core_Camera_setname, "Camera manufacturer and name", NULL},
    {"width", (getter) svs_core_Camera_getwidth, (setter) svs_core_Camera_setwidth, "Image width", NULL},
    {"height", (getter) svs_core_Camera_getheight, (setter) svs_core_Camera_setheight, "Image height", NULL},
    {"pixelclock", (getter) svs_core_Camera_getpixelclock, (setter) svs_core_Camera_setpixelclock, "Pixel Clock of camera", NULL},
    {"color_mode", (getter) svs_core_Camera_getcolor_mode, (setter) svs_core_Camera_setcolor_mode,
        "Color mode of images.\n\n"
        "It is recommended to change color mode only when not\n"
        "capturing images, and to free and reallocate memory\n"
        "after changing, as the new color mode may have a different\n"
        "bit depth.", NULL},
    {"gain", (getter) svs_core_Camera_getgain, (setter) svs_core_Camera_setgain, "Hardware gain (individual RGB gains not yet supported)", NULL},
    {"exposure", (getter) svs_core_Camera_getexposure, (setter) svs_core_Camera_setexposure, "Exposure time in milliseconds", NULL},
    {"auto_exposure", (getter) svs_core_Camera_getauto_exposure, (setter) svs_core_Camera_setauto_exposure, "Auto exposure/gain", NULL},
    {"auto_exposure_brightness", (getter) svs_core_Camera_getauto_exposure_brightness, (setter) svs_core_Camera_setauto_exposure_brightness, "Auto exposure reference brightness (0 to 1)", NULL},
    {"auto_speed", (getter) svs_core_Camera_getauto_speed, (setter) svs_core_Camera_setauto_speed, "Auto speed", NULL},
    {"auto_white_balance", (getter) svs_core_Camera_getauto_white_balance, (setter) svs_core_Camera_setauto_white_balance, "Auto White Balance", NULL},
    {"color_correction", (getter) svs_core_Camera_getcolor_correction, (setter) svs_core_Camera_setcolor_correction, "IR color correction factor", NULL},
    {"continuous_capture", (getter) svs_core_Camera_getcontinuous_capture, (setter) svs_core_Camera_setcontinuous_capture,
        "Enable or disable camera continuous capture (free-run) mode.\n\n"
        "Once set to True, continuous capture is enabled, and methods\n"
        "to retrieve images can be called.", NULL},
    {NULL}
};
