# -*- coding: utf-8 -*-

"""Main module."""

def hello_world():
    """Say hello to world.

    :returns: Nothing
    :rtype: NoneType

    """
    print("Hello world")


def documentation_example(a, b):
    """This sentence briefly describes the function.

    For more information on docstrings, see:

        https://stackoverflow.com/a/24385103

    This function returns a tuple containing the input parameters.

    :param a: this is a first parameter (int)
    :param b: this is the second parameter (string)
    :returns: a tuple
    :rtype: (int, string)

    """
    return (a,b)
