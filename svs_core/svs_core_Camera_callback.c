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

#define PY_ARRAY_UNIQUE_SYMBOL  svs_core_ARRAY_API
#define NO_IMPORT_ARRAY

#include <Python.h>
#include <datetime.h>
#include <numpy/arrayobject.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <sys/queue.h>
#include <sys/time.h>
#include <time.h>
#include <libsvgige/svgige.h>
#include "svs_core.h"

void import_datetime(void) {
    PyDateTime_IMPORT;
}

/*
 * Asynchronously raise an exception in the main thread.
 *
 * Since the callbacks are called from a different thread, there is not
 * a good way to indicate errors to the program.
 *
 * In the event of fatal problem, raise SVSAsyncError in the main thread.
 * Note that this exception will probably not be caught, and should only be
 * used when nothing else can be done.
 */
static void raise_asynchronous_exception(svs_core_Camera *self) {
    PyThreadState_SetAsyncExc(self->main_thread->thread_id, SVSAsyncError);

}

static void timestamp_to_timeval(svs_core_Camera *self, uint64_t timestamp,
                                 struct timeval *tv) {
    double timestamp_sec = ((double)timestamp)/self->tick_frequency;

    tv->tv_sec = (int) timestamp_sec;
    tv->tv_usec = (int) round(1e6 * (timestamp_sec - tv->tv_sec));
}

/*
 * Determine camera boot time, return in boot.
 *
 * This function attempts to synchronize the camera's internal timestamps
 * with the system time, and there be be some small error.
 *
 * @param self  Camera object
 * @param boot  Camera boot time returned here
 * @returns 0 on success, negative on error, and sets asynchronous exception
 */
static int camera_boot_time(svs_core_Camera *self, struct timeval *boot) {
    uint64_t ticks;
    struct timeval system, camera_delta;
    int ret;

    /* Current time - time since camera boot = camera boot time */

    gettimeofday(&system, NULL);

    ret = Camera_getTimestampCounter(self->handle, &ticks);
    if (ret != SVGigE_SUCCESS) {
        raise_asynchronous_exception(self);
        return -1;
    }

    timestamp_to_timeval(self, ticks, &camera_delta);
    timersub(&system, &camera_delta, boot);

    return 0;
}

/*
 * Create a DateTime object of the time when the image was captured.
 *
 * @param self      Camera object
 * @param svimage   Image
 * @returns DateTime object, or NULL on error and sets asynchronous exception
 */
static PyObject *image_timestamp(svs_core_Camera *self, SVGigE_IMAGE *svimage) {
    struct timeval camera_boot, image_delta, image_timestamp;
    struct tm timestamp;
    PyObject *image_datetime;
    int ret;

    ret = camera_boot_time(self, &camera_boot);
    if (ret) {
        return NULL;
    }

    /* Image time since boot */
    timestamp_to_timeval(self, svimage->Timestamp, &image_delta);

    /* Camera boot time + image time from camera boot = image time */
    timeradd(&camera_boot, &image_delta, &image_timestamp);

    gmtime_r(&image_timestamp.tv_sec, &timestamp);

    image_datetime = PyDateTime_FromDateAndTime(timestamp.tm_year + 1900,
            timestamp.tm_mon + 1, timestamp.tm_mday, timestamp.tm_hour,
            timestamp.tm_min, timestamp.tm_sec, image_timestamp.tv_usec);

    return image_datetime;
}

/*
 * Create a dictionary with various image metadata.
 *
 * @param self      Camera object
 * @param svimage   Image
 * @returns dict object, or NULL on error and sets asynchronous exception
 */
static PyObject *image_info(svs_core_Camera *self, SVGigE_IMAGE *svimage) {
    PyObject *dict, *timestamp;

    dict = PyDict_New();
    if (!dict) {
        raise_asynchronous_exception(self);
        return NULL;
    }

    timestamp = image_timestamp(self, svimage);
    if (!timestamp) {
        Py_DECREF(dict);
        return NULL;
    }

    PyObject *width = Py_BuildValue("i", svimage->ImageWidth);
    PyObject *height = Py_BuildValue("i", svimage->ImageHeight);
    PyObject *image_count = Py_BuildValue("i", svimage->ImageCount);
    PyObject *frame_loss = Py_BuildValue("i", svimage->FrameLoss);
    PyObject *packet_count = Py_BuildValue("i", svimage->PacketCount);
    PyObject *packet_resend = Py_BuildValue("i", svimage->PacketResend);
    PyObject *transfer_time = Py_BuildValue("i", svimage->TransferTime);

    PyDict_SetItemString(dict, "timestamp", timestamp);
    PyDict_SetItemString(dict, "width", width);
    PyDict_SetItemString(dict, "height", height);
    PyDict_SetItemString(dict, "image_count", image_count);
    PyDict_SetItemString(dict, "frame_loss", frame_loss);
    PyDict_SetItemString(dict, "packet_count", packet_count);
    PyDict_SetItemString(dict, "packet_resend", packet_resend);
    PyDict_SetItemString(dict, "transfer_time", transfer_time);

    Py_DECREF(timestamp);
    Py_DECREF(width);
    Py_DECREF(height);
    Py_DECREF(image_count);
    Py_DECREF(frame_loss);
    Py_DECREF(packet_count);
    Py_DECREF(packet_resend);
    Py_DECREF(transfer_time);

    return dict;
}

static PyObject *image_array(svs_core_Camera *self, SVGigE_IMAGE *svimage) {
    npy_intp dims[2] = {svimage->ImageHeight, svimage->ImageWidth};
    PyArrayObject *array;
    int ret, pixel_size, numpy_type, numpy_size, convert;

    pixel_size = svimage->PixelType & GVSP_PIX_EFFECTIVE_PIXELSIZE_MASK;

    switch (pixel_size) {
    case GVSP_PIX_OCCUPY8BIT:
        numpy_type = NPY_UINT8;
        numpy_size = dims[0]*dims[1];
        convert = 0;
        break;
    case GVSP_PIX_OCCUPY12BIT:
        numpy_type = NPY_UINT16;
        numpy_size = 2*dims[0]*dims[1];
        convert = 1;
        break;
    case GVSP_PIX_OCCUPY16BIT:
        numpy_type = NPY_UINT16;
        numpy_size = 2*dims[0]*dims[1];
        convert = 0;
        break;
    default:
        return NULL;
    }

    array = (PyArrayObject*)PyArray_SimpleNew(2, dims, numpy_type);
    if (!array) {
        raise_asynchronous_exception(self);
        return NULL;
    }

    if (convert) {
        ret = Image_getImage12bitAs16bit(svimage->ImageData, dims[1], dims[0],
                svimage->PixelType, PyArray_DATA(array), 2*dims[0]*dims[1]);
        if (ret) {
            raise_asynchronous_exception(self);
            Py_DECREF((PyObject*)array);
            return NULL;
        }
    }
    else {
        memcpy(PyArray_DATA(array), svimage->ImageData, numpy_size);
    }

    return (PyObject *) array;
}

/*
 * New image handler
 *
 * Called by the stream callback to handle new images.  Converts them
 * to the appropriate type and collects data about them, then adds to
 * the image queue.
 */
static SVGigE_RETURN svs_core_Camera_new_image(svs_core_Camera *self,
                                               SVGigE_SIGNAL *signal) {
    SVGigE_IMAGE *svimage = signal->Data;
    struct image *image;
    PyGILState_STATE gstate;

    image = malloc(sizeof(*image));
    if (!image) {
        return SVGigE_OUT_OF_MEMORY;
    }

    /* Grab the GIL */
    gstate = PyGILState_Ensure();

    image->info = image_info(self, svimage);
    if (!image->info) {
        PyGILState_Release(gstate);
        goto err_free_image;
    }

    image->array = image_array(self, svimage);
    if (!image->array) {
        PyGILState_Release(gstate);
        goto err_decref_info;
    }

    /* Queue full, drop the first item */
    if (self->images_max && self->images_length == self->images_max) {
        struct image *image = TAILQ_FIRST(&self->images);
        TAILQ_REMOVE(&self->images, image, entry);
        Py_DECREF(image->array);
        Py_DECREF(image->info);
        self->images_length--;
    }

    /* Add image to queue */
    TAILQ_INSERT_TAIL(&self->images, image, entry);
    self->images_length++;

    /* Release the GIL */
    PyGILState_Release(gstate);

    return SVGigE_SUCCESS;

err_decref_info:
    Py_DECREF(image->info);
err_free_image:
    free(image);
    return SVGigE_ERROR;
}

SVGigE_RETURN svs_core_Camera_stream_callback(SVGigE_SIGNAL *signal,
                                              void *context) {
    svs_core_Camera *self = context;
    int ret;

    switch (signal->SignalType) {
    case SVGigE_SIGNAL_FRAME_COMPLETED:
        ret = svs_core_Camera_new_image(self, signal);
        break;
    default:
        ret = SVGigE_SUCCESS;
    }

    return ret;
}
