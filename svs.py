# Copyright (c) 2013, North Carolina State University Aerial Robotics Club
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the North Carolina State University Aerial Robotics Club
#       nor the names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import svs_core
from svs_core import camera_list

class Camera(svs_core.Camera):
    """
    SVS Camera object

    Provides access to, and control of, SVS-VISTEK GigE machine vision cameras.
    This class provides as attributes many of the camera settings.  It handles
    image capture internally, and provides methods to get images from the
    camera.

    If ip and source_ip are not passed in, the class connects to the first
    available camera.

    Arguments:
        logger (optional): logging object to use for log output.

        These are optional, but both are required if either is passed:
            ip: IP address of camera to connect to.
            source_ip: IP address of local interface used for connection

        buffer_count (optional): Number of internal buffers for SVGigE
            streaming channels.
        packet_size (optional): MTU packet size.
        queue_length (optional): Maximum number of images to queue for
            return by next().  Once this limit is reached, old images are
            dropped from the queue.  A length of zero allows infinite
            images to queue.
    """

    def __init__(self, *args, **kwargs):
        logging.basicConfig()   # Configure logging, if it isn't already
        self.logger = kwargs.pop('logger', None) or logging.getLogger(__name__)

        if not 'ip' in kwargs or not 'local_ip' in kwargs:
            cameras = camera_list()
            if len(cameras) == 0:
                raise IOError("No cameras found")
            kwargs['ip'] = cameras[0]['ip']
            kwargs['source_ip'] = cameras[0]['local_ip']

        super(Camera, self).__init__(*args, **kwargs)


def number_cameras():
    """
    Determines total number of cameras available.
    """
    return len(camera_list())
