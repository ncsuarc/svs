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
#include <datetime.h>
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
 * @returns 0 on success, negative on error
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
 * Return a DateTime object of the time when the image was captured.
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

    image->timestamp = image_timestamp(self, svimage);
    if (!image->timestamp) {
        return SVGigE_ERROR;
    }

    /* Add image to queue */
    TAILQ_INSERT_TAIL(&self->images, image, entry);

    /* Release the GIL */
    PyGILState_Release(gstate);

    return SVGigE_SUCCESS;
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
