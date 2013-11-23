#ifndef SVS_CORE_H_INCLUDED
#define SVS_CORE_H_INCLUDED

#include <pthread.h>
#include <sys/queue.h>
#include <libsvgige/svgige.h>

/* Module methods */
extern PyMethodDef svs_coreMethods[];

struct image {
    PyObject *array;
    PyObject *info;
    TAILQ_ENTRY(image) entry;
};

/* Camera class */
typedef struct {
    PyObject_HEAD;
    /* Externally available elements (in Python) */
    uint32_t        width;
    uint32_t        height;
    /* Private elements */
    int             ready;
    Camera_handle   handle;
    Stream_handle   stream;
    unsigned int    stream_ip;
    unsigned short  stream_port;
    int             depth;
    unsigned int    buffer_size;
    uint64_t        tick_frequency;
    PyObject        *name;
    TAILQ_HEAD(image_head, image) images;   /* Locked with the GIL */
    unsigned int    images_length;          /* Queue length */
    unsigned int    images_max;             /* Max queue length */
    PyThreadState   *main_thread;
} svs_core_Camera;   /* Be sure to update svs_core_Camera_members with new entries */

enum ready {
    NOT_READY,
    CONNECTED,
    NAME_ALLOCATED,
    READY,
};

/*
 * Add constants to module
 *
 * Add the necessary SVGigE constants to the module
 */
void add_constants(PyObject *m);

/*
 * Raise an exception for an unknown SVS error code.
 *
 * Attempts to lookup error message with getErrorMessage(),
 * then raises an SVSError with the error code and message.
 *
 * @param error Error code returned from SVS function
 */
void raise_general_error(int error);

extern PyTypeObject svs_core_CameraType;
extern PyMethodDef svs_core_Camera_methods[];
extern PyGetSetDef svs_core_Camera_getseters[];

/* SVS Exceptions */
extern PyObject *SVSError;
extern PyObject *SVSAsyncError;
extern PyObject *SVSNoImagesError;

/*
 * Camera stream callback
 *
 * Callback for image stream.  This function is called for all new images.
 */
SVGigE_RETURN svs_core_Camera_stream_callback(SVGigE_SIGNAL *signal,
                                              void *context);

/*
 * Import datetime module
 *
 * This module is needed in svs_core_Camera_callback.c, but needs to be
 * imported on a per-file basis, so provide a function to call from the
 * main initialization.
 */
void import_datetime(void);

/* Utility functions */

/*
 * Convert an IP address string to an int representation
 *
 * Takes a string in the form "127.0.0.1", and returns the
 * integer representation of the address, 0x7F000001.
 */
uint32_t ip_string_to_int(const char *ip);

/*
 * Convert an IP address int to string representation
 *
 * Takes an unsigned int of the IP address (each byte represents an octet),
 * and writes the string format into the buffer.
 *
 * The buffer must be at least 16 bytes.
 */
void ip_int_to_string(uint32_t ip, char buf[16]);

#endif
