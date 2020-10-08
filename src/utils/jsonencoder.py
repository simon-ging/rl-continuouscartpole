"""More beautiful JSON encoder than the default json.dump.
Does indents and newlines but keeps lists on single line.
"""

import numpy
import collections


def to_json(obj, lvl=0, indent=4, space=" ", newline="\n"):
    json_str = ""
    if isinstance(obj, collections.Mapping):
        json_str += "{" + newline
        comma = ""
        for key, val in obj.items():
            json_str += comma
            comma = ",\n"
            json_str += space * indent * (lvl + 1)
            json_str += '"' + str(key) + '":' + space
            json_str += to_json(val, lvl + 1)

        json_str += newline + space * indent * lvl + "}"
    elif isinstance(obj, str):
        json_str += '"' + obj + '"'
    elif isinstance(obj, collections.Iterable):
        json_str += "[" + ",".join([to_json(e, lvl + 1) for e in obj]) + "]"
    elif isinstance(obj, bool):
        json_str += "true" if obj else "false"
    elif isinstance(obj, int):
        json_str += str(obj)
    elif isinstance(obj, float):
        json_str += '%.7g' % obj
    elif isinstance(obj, numpy.ndarray) and numpy.issubdtype(
            obj.dtype, numpy.integer):
        json_str += "[" + ','.join(map(str, obj.flatten().tolist())) + "]"
    elif isinstance(obj, numpy.ndarray) and numpy.issubdtype(
            obj.dtype, numpy.inexact):
        json_str += "[" + ','.join(
            map(lambda x: '%.7g' % x, obj.flatten().tolist())) + "]"
    elif obj is None:
        json_str += 'null'
    else:
        raise TypeError(
            "Unknown type '%s' for json serialization" % str(type(obj)))
    return json_str
