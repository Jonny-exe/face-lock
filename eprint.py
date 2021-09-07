#!/usr/bin/python3
import re
import traceback, sys

from inspect import getsourcefile
from os.path import abspath


def eprint(value):
    p = re.compile("(?<=eprint\()[\D\W\S]*(?=\))")
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    line = stack[0].line
    result = p.search(line)
    output = f"{result[0]}: {value}"
    print(output)
    return result


if __name__ == "__main__":
    hello = 1
    eprint(hello)
