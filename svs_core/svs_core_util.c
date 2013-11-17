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
#include <stdint.h>
#include <inttypes.h>
#include "svs_core.h"

uint32_t ip_string_to_int(const char *ip) {
    uint8_t octet[4];

    sscanf(ip, "%" SCNu8 ".%" SCNu8 ".%" SCNu8 ".%" SCNu8, &octet[3], &octet[2],
           &octet[1], &octet[0]);

    return (octet[3] << 24) | (octet[2] << 16) | (octet[1] << 8) | octet[0];
}

void ip_int_to_string(uint32_t ip, char buf[16]) {
    uint8_t octet[4];

    octet[0] = ip & 0xff;
    octet[1] = (ip >> 8) & 0xff;
    octet[2] = (ip >> 16) & 0xff;
    octet[3] = (ip >> 24) & 0xff;

    snprintf(buf, 16, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8, octet[3],
             octet[2], octet[1], octet[0]);
}
