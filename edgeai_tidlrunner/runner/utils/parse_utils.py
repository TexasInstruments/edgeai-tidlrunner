# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import ast


def str_to_dict(v):
    if v is None:
        return None
    #
    if isinstance(v, str):
        v = v.replace(' ', '')
        v = v.split(',')
    #
    d = dict()
    for word in v:
        key, value = word.split(':')
        d.update({key:value})
    #
    return d


def str_to_int(v):
    if v in ('', None, 'None', 'none', 'null'):
        return None
    else:
        return int(v)


def str_to_bool(v):
    if v is None:
        return False
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null', 'false', 'no', '0'):
            return False
        elif v.lower() in ('true', 'yes', '1'):
            return True
        #
    #
    return bool(v)


def int_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null', 'false', 'no'):
            return None
        elif v.lower() in ('0',):
            return 0
        elif v.lower() in ('true', 'yes', '1'):
            return 1
        #
    #
    return int(v)


def float_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null', 'false', 'no'):
            return None
        elif v.lower() in ('0',):
            return 0.0
        elif v.lower() in ('true', 'yes', '1'):
            return 1.0
        #
    #
    return float(v)


def str_or_none(v):
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null'):
            return None
        #
    #
    return str(v)


def str_or_none_or_bool(v):
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() in ('', 'none', 'null'):
            return None
        elif v.lower() in ('false', 'no', '0'):
            return False
        elif v.lower() in ('true', 'yes', '1'):
            return True
        #
    #
    return str(v)


def to_int(v):
    try:
        v = int(v)
        return v
    except ValueError:
        return None


def to_float(v):
    try:
        v = float(v)
        return v
    except ValueError:
        return None


def is_number(v):
    return to_float(v) is not None


def int_or_tuple(v):
    if ',' in v:
        v = tuple(map(to_int, v.split(',')))
    else:
        v = to_int(v)
    #
    return v


def float_or_tuple(v):
    if ',' in v:
        v = tuple(map(to_float, v.split(',')))
    else:
        v = to_int(v)
    #
    return v


def str_to_list_of_tuples(v):
    if isinstance(v, str):
        v = ast.literal_eval(v)
    #
    assert isinstance(v, list), f'ERROR: invalid parsing output: {v}'
    return v


def str_to_literal(v):
    lst = ast.literal_eval(v)
    return lst

