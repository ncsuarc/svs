SVS Python Module
==================

A module for interfacing with SVS-VISTEK GigE machine vision cameras.
This module wraps the SVGigE SDK, providing a convient Python interface
to much of the SDK.  The svs.Camera object provides attributes for easily
controlling camera settings and capturing images.

## Requirements

The ids module is written in Python and C, using the Python C API, and supports
both Python 2 and Python 3.  It has been tested using Python 2.7 and Python 3.2.

The module has only been tested on Linux, and likely does not support Windows.

Build requirements:

* SVGigE SDK
    * SDK version 1.4.19.51-4 Beta or higher is supported
    * The SDK can be acquired from 
        [SVS-VISTEK](http://www.svs-vistek.com/intl/en/index.php)
    * The SDK should be installed in /usr/lib/ or /usr/local/lib/

Known working cameras:

* SVS-VISTEK GmbH SVS11002CSGEV2

## Building and installing

Once all dependencies are met, it is simple to build and install the module:

    $ python setup.py install

Or, for Python 3:

    $ python3 setup.py install

Of course, if the installation location requires root permissions, `sudo` may
be necessary.

## Usage

### Capturing images

The svs module makes it easy to control a camera.  Just initialize the Camera
object and set the attributes of interest, then start image capture.

    >>> import svs
    >>> cam = svs.Camera()
    >>> cam.framerate = 1               # Capture 1 image per second
    >>> cam.exposure = 50               # Exposure time in milliseconds
    >>> cam.continuous_capture = True   # Start image capture

Once image capture is started, images will be queued up in the background.
Use the next() method to grab the first image from the queue.  The maximum
number of images in the queue is specified with the `queue_length` argument
to the Camera object.  When the queue is full, old images will be dropped
to make room for new ones.  next() will raise SVSNoImagesError if there are
no images currently available.

    >>> img, meta = cam.next()

When finished capturing images, stop continuous capture.

    >>> cam.continuous_capture = False

You may wish to call next() until it raises SVSNoImagesError to flush the queue
of remaining images.

### Working with images

The image returned by the next() method is a Numpy array containing the image
data, either monochrome, or Bayer data, depending on the camera.  Several
different modules can be used to manipulate and save the images.

The [tiffutils](http://github.com/ncsuarc/tiffutils) module provides a method
to save the images as valid DNG files, useful for saving raw Bayer images.

    >>> import tiffutils
    >>> tiffutils.save_dng(img, "test.dng", camera=cam.name,
                           cfa_pattern=tiffutils.CFA_GRBG)

OpenCV can be used to perform Bayer interpolation, for further data processing
or saving in a more traditional format.

    >>> import cv2
    >>> rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
    >>> cv2.imwrite('cv2.png', rgb)

If saving images as JPEG, it is important to note that cv2.imwrite expects an
8-bit Numpy array, but the next() method will return a 16-bit Numpy array, so
it needs to be converted before saving.

    >>> import numpy as np
    >>> rgb8 = np.right_shift(rgb, 8)
    >>> cv2.imwrite('cv2.jpg', rgb8)
