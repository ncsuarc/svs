SVS Python Module
==================

**This module is incomplete, and does not currently capture images.**

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

## Building and installing

Once all dependencies are met, it is simple to build and install the module:

    $ python setup.py install

Or, for Python 3:

    $ python3 setup.py install

Of course, if the installation location requires root permissions, `sudo` may
be necessary.

## Usage

The svs module makes it easy to control a camera.  Just initialize the Camera
object and set the attributes of interest, then start image capture.

    >>> import svs
    >>> cam = ids.Camera()
    >>> cam.continuous_capture = True               # Start image capture

You will be able to do more soon...
