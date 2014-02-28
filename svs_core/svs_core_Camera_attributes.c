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
    PyObject *dict = PyDict_New();
    if (!dict) {
        return NULL;
    }

    PyObject *manufacturer = Py_BuildValue("s",
            Camera_getManufacturerName(self->handle));
    PyObject *model = Py_BuildValue("s",
            Camera_getModelName(self->handle));
    PyObject *device_version = Py_BuildValue("s",
            Camera_getDeviceVersion(self->handle));
    PyObject *specific_information = Py_BuildValue("s",
            Camera_getManufacturerSpecificInformation(self->handle));
    PyObject *serial_number = Py_BuildValue("s",
            Camera_getSerialNumber(self->handle));
    PyObject *user_name = Py_BuildValue("s",
            Camera_getUserDefinedName(self->handle));
    PyObject *ip = Py_BuildValue("s",
            Camera_getIPAddress(self->handle));
    PyObject *subnet_mask = Py_BuildValue("s",
            Camera_getSubnetMask(self->handle));
    PyObject *mac_address = Py_BuildValue("s",
            Camera_getMacAddress(self->handle));

    PyDict_SetItemString(dict, "manufacturer", manufacturer);
    PyDict_SetItemString(dict, "model", model);
    PyDict_SetItemString(dict, "device_version", device_version);
    PyDict_SetItemString(dict, "specific_information", specific_information);
    PyDict_SetItemString(dict, "serial_number", serial_number);
    PyDict_SetItemString(dict, "user_name", user_name);
    PyDict_SetItemString(dict, "ip", ip);
    PyDict_SetItemString(dict, "subnet_mask", subnet_mask);
    PyDict_SetItemString(dict, "mac_address", mac_address);

    Py_DECREF(manufacturer);
    Py_DECREF(model);
    Py_DECREF(device_version);
    Py_DECREF(specific_information);
    Py_DECREF(serial_number);
    Py_DECREF(user_name);
    Py_DECREF(ip);
    Py_DECREF(subnet_mask);
    Py_DECREF(mac_address);

    return dict;
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

static PyObject *svs_core_Camera_getgain(svs_core_Camera *self, void *closure) {
    float gain;
    int ret;

    ret = Camera_getGain(self->handle, &gain);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    return PyFloat_FromDouble(gain);
}

static int svs_core_Camera_setgain(svs_core_Camera *self, PyObject *value, void *closure) {
    int ret;
    float gain;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'gain'");
        return -1;
    }

    gain = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    ret = Camera_setGain(self->handle, gain);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
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
    float exposure;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'exposure'");
        return -1;
    }

    exposure = PyFloat_AsDouble(value);   /* Exposure in milliseconds */
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

static PyObject *svs_core_Camera_getauto_exposure_min(svs_core_Camera *self, void *closure) {
    float min, max;
    int ret;

    ret = Camera_getAutoExposureLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    /* SVS gives us microseconds, return milliseconds */
    return PyFloat_FromDouble(min/1000);
}

static int svs_core_Camera_setauto_exposure_min(svs_core_Camera *self, PyObject *value, void *closure) {
    float min, max;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure_min'");
        return -1;
    }

    /* Get current min/max */
    ret = Camera_getAutoExposureLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    /* Python gives us milliseconds, give SVS microseconds */
    min = 1000*PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    /* Set new min, current max */
    ret = Camera_setAutoExposureLimits(self->handle, min, max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getauto_exposure_max(svs_core_Camera *self, void *closure) {
    float min, max;
    int ret;

    ret = Camera_getAutoExposureLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    /* SVS gives us microseconds, return milliseconds */
    return PyFloat_FromDouble(max/1000);
}

static int svs_core_Camera_setauto_exposure_max(svs_core_Camera *self, PyObject *value, void *closure) {
    float min, max;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure_max'");
        return -1;
    }

    /* Get current min/max */
    ret = Camera_getAutoExposureLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    /* Python gives us milliseconds, give SVS microseconds */
    max = 1000*PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    /* Set new max, current min */
    ret = Camera_setAutoExposureLimits(self->handle, min, max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getauto_gain_min(svs_core_Camera *self, void *closure) {
    float min, max;
    int ret;

    ret = Camera_getAutoGainLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    return PyFloat_FromDouble(min);
}

static int svs_core_Camera_setauto_gain_min(svs_core_Camera *self, PyObject *value, void *closure) {
    float min, max;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_gain_min'");
        return -1;
    }

    /* Get current min/max */
    ret = Camera_getAutoGainLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    min = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    /* Set new min, current max */
    ret = Camera_setAutoGainLimits(self->handle, min, max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getauto_gain_max(svs_core_Camera *self, void *closure) {
    float min, max;
    int ret;

    ret = Camera_getAutoGainLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    return PyFloat_FromDouble(max);
}

static int svs_core_Camera_setauto_gain_max(svs_core_Camera *self, PyObject *value, void *closure) {
    float min, max;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_gain_max'");
        return -1;
    }

    /* Get current min/max */
    ret = Camera_getAutoGainLimits(self->handle, &min, &max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    max = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    /* Set new max, current min */
    ret = Camera_setAutoGainLimits(self->handle, min, max);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getauto_exposure_brightness(svs_core_Camera *self, void *closure) {
    float brightness;
    int ret;

    ret = Camera_getAutoGainBrightness(self->handle, &brightness);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    /* Brightness in range 0..255 */
    return PyFloat_FromDouble(brightness/255);
}

static int svs_core_Camera_setauto_exposure_brightness(svs_core_Camera *self, PyObject *value, void *closure) {
    float brightness;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure_brightness'");
        return -1;
    }

    /* Brightness in range 0..255 */
    brightness = 255*PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    ret = Camera_setAutoGainBrightness(self->handle, brightness);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getauto_exposure_dynamics(svs_core_Camera *self, void *closure) {
    float i, d;
    int ret;

    ret = Camera_getAutoGainDynamics(self->handle, &i, &d);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    return Py_BuildValue("(ff)", i, d);
}

static int svs_core_Camera_setauto_exposure_dynamics(svs_core_Camera *self, PyObject *value, void *closure) {
    float i, d;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure_dynamics'");
        return -1;
    }

    /* Provide nicer exceptions for bad type/tuple vs ParseTuple */
    if (!PyTuple_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Value should be a tuple in the form (I,D)");
        return -1;
    }

    if (PyTuple_Size(value) != 2) {
        PyErr_SetString(PyExc_ValueError, "Value should be a tuple in the form (I,D)");
        return -1;
    }

    if (!PyArg_ParseTuple(value, "ff", &i, &d)) {
        return -1;
    }

    ret = Camera_setAutoGainDynamics(self->handle, i, d);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
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

static PyObject *svs_core_Camera_getframerate(svs_core_Camera *self, void *closure) {
    float framerate;
    int ret;

    ret = Camera_getFrameRate(self->handle, &framerate);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    return PyFloat_FromDouble(framerate);
}

static int svs_core_Camera_setframerate(svs_core_Camera *self, PyObject *value, void *closure) {
    float framerate;
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'framerate'");
        return -1;
    }

    framerate = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }

    ret = Camera_setFrameRate(self->handle, framerate);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return -1;
    }

    return 0;
}

static PyObject *svs_core_Camera_getactual_framerate(svs_core_Camera *self, void *closure) {
    float framerate;
    int ret;

    ret = StreamingChannel_getActualFrameRate(self->stream, &framerate);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    return PyFloat_FromDouble(framerate);
}

static int svs_core_Camera_setactual_framerate(svs_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_TypeError, "Cannot modify attribute 'actual_framerate'");
    return -1;
}

PyGetSetDef svs_core_Camera_getseters[] = {
    {"info", (getter) svs_core_Camera_getinfo, (setter) svs_core_Camera_setinfo, "Camera info", NULL},
    {"name", (getter) svs_core_Camera_getname, (setter) svs_core_Camera_setname, "Camera manufacturer and name", NULL},
    {"width", (getter) svs_core_Camera_getwidth, (setter) svs_core_Camera_setwidth, "Image width", NULL},
    {"height", (getter) svs_core_Camera_getheight, (setter) svs_core_Camera_setheight, "Image height", NULL},
    {"pixelclock", (getter) svs_core_Camera_getpixelclock, (setter) svs_core_Camera_setpixelclock, "Pixel Clock of camera", NULL},
    {"gain", (getter) svs_core_Camera_getgain, (setter) svs_core_Camera_setgain, "Camera gain (0..18dB)", NULL},
    {"exposure", (getter) svs_core_Camera_getexposure, (setter) svs_core_Camera_setexposure, "Exposure time in milliseconds", NULL},
    {"auto_exposure", (getter) svs_core_Camera_getauto_exposure, (setter) svs_core_Camera_setauto_exposure, "Auto exposure/gain (exposure adjusted to limit, then gain used to limit)", NULL},
    {"auto_exposure_min", (getter) svs_core_Camera_getauto_exposure_min, (setter) svs_core_Camera_setauto_exposure_min, "Minimum exposure for auto exposure (ms)", NULL},
    {"auto_gain_min", (getter) svs_core_Camera_getauto_gain_min, (setter) svs_core_Camera_setauto_gain_min, "Minimum gain for auto gain (dB)", NULL},
    {"auto_gain_max", (getter) svs_core_Camera_getauto_gain_max, (setter) svs_core_Camera_setauto_gain_max, "Maximum gain for auto gain (dB)", NULL},
    {"auto_exposure_max", (getter) svs_core_Camera_getauto_exposure_max, (setter) svs_core_Camera_setauto_exposure_max, "Maximum exposure for auto exposure (ms)", NULL},
    {"auto_exposure_brightness", (getter) svs_core_Camera_getauto_exposure_brightness, (setter) svs_core_Camera_setauto_exposure_brightness, "Auto exposure reference brightness (0 to 1)", NULL},
    {"auto_exposure_dynamics", (getter) svs_core_Camera_getauto_exposure_dynamics, (setter) svs_core_Camera_setauto_exposure_dynamics,
        "Auto exposure dynamics (I, D)\n\n"
        "I and D parameters used by the auto exposure PID\n"
        "controller for adjusting exposure and gain to achieve the desired\n"
        "brightness.\n\n"
        "In the form of a tuple with the following values:\n"
        "   I: Integral PID parameter\n"
        "   D: Derivative PID parameter\n\n", NULL},
    {"continuous_capture", (getter) svs_core_Camera_getcontinuous_capture, (setter) svs_core_Camera_setcontinuous_capture,
        "Enable or disable camera continuous capture (free-run) mode.\n\n"
        "Once set to True, continuous capture is enabled, and methods\n"
        "to retrieve images can be called.", NULL},
    {"framerate", (getter) svs_core_Camera_getframerate, (setter) svs_core_Camera_setframerate,
        "Desired image capture framerate\n\n"
        "Actual framerate may be slower than this value.\n"
        "See actual_framerate for measured framerate.", NULL},
    {"actual_framerate", (getter) svs_core_Camera_getactual_framerate, (setter) svs_core_Camera_setactual_framerate,
        "Actual measured image capture framerate\n\n"
        "Actual achieved framerate, based on measurement of received images.\n"
        "Will not be valid until image capture is active.", NULL},
    {NULL}
};
