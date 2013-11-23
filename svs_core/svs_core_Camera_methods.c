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
#include <sys/queue.h>
#include <libsvgige/svgige.h>
#include "svs_core.h"

static PyObject *svs_core_Camera_close(svs_core_Camera *self, PyObject *args, PyObject *kwds) {
    int ret;

    ret = closeStream(self->stream);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    ret = closeCamera(self->handle);
    if (ret != SVGigE_SUCCESS) {
        raise_general_error(ret);
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *svs_core_Camera_next(svs_core_Camera *self, PyObject *args, PyObject *kwds) {
    struct image *image;
    PyObject *ret;

    if (TAILQ_EMPTY(&self->images)) {
        PyErr_SetString(SVSNoImagesError, "No images available");
        return NULL;
    }

    image = TAILQ_FIRST(&self->images);
    TAILQ_REMOVE(&self->images, image, entry);
    self->images_length--;

    ret = Py_BuildValue("(OO)", image->array, image->info);

    Py_DECREF(image->array);
    Py_DECREF(image->info);

    free(image);

    return ret;
}

PyMethodDef svs_core_Camera_methods[] = {
    {"close", (PyCFunction) svs_core_Camera_close, METH_NOARGS,
        "close()\n\n"
        "Closes open camera.\n\n"
        "Raises:\n"
        "    SVSError: An unknown error occured in the SVGigE SDK."
    },
    {"next", (PyCFunction) svs_core_Camera_next, METH_VARARGS,
        "next() -> image, metadata\n\n"
        "Gets next available image.\n\n"
        "Gets the next available image from the camera as a Numpy array\n"
        "Blocks until image is available, or timeout occurs.\n\n"
        "Returns:\n"
        "    (image, metadata) tuple, where image is a Numpy array containing\n"
        "    the image, and metadata is a dictionary containing image metadata.\n\n"
        "Raises:\n"
        "    SVSNoImagesError: No images are available in the queue."
    },
    {NULL}
};
