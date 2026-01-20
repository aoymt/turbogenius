#!python -u
# -*- coding: utf-8 -*-

"""

pyturbo: utilities

Todo:
    * docstrings are not completed.

"""

from __future__ import print_function

# python modules
import os
import shutil
import platform
import subprocess
import linecache
import numpy as np
from scipy.io import FortranFile
from logging import getLogger, StreamHandler, Formatter

# python special module
from pymatgen.core.periodic_table import Element, ElementBase

# turbogenius module
from .env import pyturbo_root

logger = getLogger("pyturbo").getChild(__name__)


def get_linenum_fort12(fort12="fort.12"):
    """
    Get the number of lines (records) in a fort.12 file.

    Parameters
    ----------
    fort12 : str, optional
        Name of the fort.12 file, by default "fort.12".

    Returns
    -------
    int
        Number of data records in the file.

    Notes
    -----
    This function reads a Fortran unformatted file and counts the number
    of records by checking the column length and reading the file structure.
    """
    # check column length of fort.12
    f = FortranFile(fort12, "r")
    a = f.read_reals(dtype="float64")
    column_length = len(a)
    f.close()
    head = ("head", "<i")
    tail = ("tail", "<i")
    dt = np.dtype([head, ("a", "<{}d".format(column_length)), tail])
    fd = open(fort12, "r")
    fort12_b = np.fromfile(fd, dtype=dt, count=-1)
    data_length = len(fort12_b)
    fd.close()
    return data_length


def return_element_symbol(atomic_number):
    """
    Return element symbol from atomic number.

    Parameters
    ----------
    atomic_number : int or float or str
        Atomic number of the element.

    Returns
    -------
    str
        Element symbol (e.g., "H", "He", "Li").
    """
    atomic_number = int(float(atomic_number))
    return str(ElementBase.from_Z(atomic_number))


def return_atomic_number(element):
    """
    Return atomic number from element symbol.

    Parameters
    ----------
    element : str
        Element symbol (e.g., "H", "He", "Li").

    Returns
    -------
    float
        Atomic number. Returns 0.0 if the element is not found.
    """
    element = str(element)
    try:
        E = Element(element)
        number = E.number
    except ValueError:
        number = 0.0
    return number


def remove_file(file):
    """
    Remove a file if it exists.

    Parameters
    ----------
    file : str
        Path to the file to remove.

    Notes
    -----
    This function silently does nothing if the file does not exist.
    """
    if os.path.isfile(file):
        os.remove(file)


def copy_file(from_file, to_file):
    """
    Copy a file from source to destination.

    Parameters
    ----------
    from_file : str
        Source file path.
    to_file : str
        Destination file path.

    Notes
    -----
    This function silently does nothing if the source and destination
    are the same file.
    """
    try:
        shutil.copy(from_file, to_file)
    except shutil.SameFileError:
        pass


def file_check(file):
    """
    Check if a file exists.

    Parameters
    ----------
    file : str
        Path to the file to check.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} is not found.")


def file_check_flag(file):
    """
    Check if a file exists and return a boolean flag.

    Parameters
    ----------
    file : str
        Path to the file to check.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    if os.path.exists(file):
        return True
    else:
        return False


def get_line_from_file(file, line_no):
    """
    Get a specific line from a file.

    Parameters
    ----------
    file : str
        Path to the file.
    line_no : int
        Line number (0-indexed) to retrieve.

    Returns
    -------
    str
        The line content at the specified line number.

    Raises
    ------
    IndexError
        If the line number is out of range.
    """
    with open(file, "r") as f:
        data = f.readlines()
    return data[line_no]


def prompt(text, checker):
    """
    Loop input() until valid input is provided.

    Parameters
    ----------
    text : str
        Prompt text to display to the user.
    checker : callable
        Function that takes the input string and returns True if valid,
        False otherwise.

    Returns
    -------
    str
        Valid user input string.

    Notes
    -----
    This function will loop indefinitely until the checker function
    returns True for the user input.
    """
    while True:
        output = input(text)
        if checker(output):
            return output


def get_str_variable_type_auto(variable):
    """
    Automatically determine the type of a string variable.

    Parameters
    ----------
    variable : str
        String variable to analyze.

    Returns
    -------
    int, float, or str
        The variable converted to the appropriate type:
        - int if the variable is a decimal integer
        - float if the variable is a decimal number (handles 'd'/'D' as 'e' for scientific notation)
        - str if the variable is not numeric (strips single quotes)

    Notes
    -----
    This function attempts to convert Fortran-style scientific notation
    (using 'd' or 'D') to Python-style (using 'e').
    """
    # logger.debug(f"variable={variable}")
    # logger.debug(f"isdecimal={variable.isdecimal()}")

    if not variable.replace("-", "").isdecimal():
        try:
            # Convert d to e in scientific notation. python supports e only
            var = variable.replace("d", "e").replace("D", "e")
            result = float(var)
            # logger.debug("This is float")
            return result
        except ValueError:
            # logger.debug("This is str")
            return str(variable).strip("'")
    else:
        # logger.debug("This is int")
        return int(variable)


def turbo_prim_orb_type_num(orb_type_chr):
    """
    Return TurboRVB primitive orbital type number from orbital type character.

    Parameters
    ----------
    orb_type_chr : str
        Orbital type character (e.g., "s", "p", "d", "f", "g", "h", "i").
        Special types: "s100", "s131", "p103", "p150".

    Returns
    -------
    int
        TurboRVB primitive orbital type number.

    Raises
    ------
    NotImplementedError
        If the orbital type character is not implemented.

    Notes
    -----
    Mapping:

    - s: 16, s100: 100, s131: 131
    - p: 36, p103: 103, p150: 150
    - d: 37
    - f: 48
    - g: 51
    - h: 72
    - i: 73
    """
    if orb_type_chr == "s":
        return 16
    elif orb_type_chr == "s100":
        return 100
    elif orb_type_chr == "s131":
        return 131
    elif orb_type_chr == "p":
        return 36
    elif orb_type_chr == "p103":
        return 103
    elif orb_type_chr == "p150":
        return 150
    elif orb_type_chr == "d":
        return 37
        # 68 -> 37 !! 12/May/2022 because both are the same in makefun.f90
        # (see. case(37,68)),
        # but 500 is usually associated with 37 (see. ioptorbcontr.f90)
    elif orb_type_chr == "f":
        return 48
    elif orb_type_chr == "g":
        return 51  # 51 or 88?
    elif orb_type_chr == "h":
        return 72
    elif orb_type_chr == "i":
        return 73
    else:
        logger.error(f"orb_type_chr={orb_type_chr} is not implemented.")
        raise NotImplementedError(f"Not implemented orb_type_chr={orb_type_chr}")


def turbo_cont_orb_type_num(orb_type_chr):
    """
    Return TurboRVB contracted orbital type number from orbital type character.

    Parameters
    ----------
    orb_type_chr : str
        Orbital type character (e.g., "s", "p", "d", "f", "g", "h", "i").

    Returns
    -------
    int
        TurboRVB contracted orbital type number.

    Raises
    ------
    NotImplementedError
        If the orbital type character is not implemented.

    Notes
    -----
    Mapping:

    - s: 300
    - p: 400
    - d: 500
    - f: 600
    - g: 700
    - h: 800
    - i: 900
    """
    if orb_type_chr == "s":
        return 300
    elif orb_type_chr == "p":
        return 400
    elif orb_type_chr == "d":
        return 500
    elif orb_type_chr == "f":
        return 600
    elif orb_type_chr == "g":
        return 700
    elif orb_type_chr == "h":
        return 800
    elif orb_type_chr == "i":
        return 900
    else:
        logger.error(f"orb_type_chr={orb_type_chr} is not implemented.")
        raise NotImplementedError(f"Not implemented orb_type_chr={orb_type_chr}")


def turbo_conv_cont_to_prim_orb_type_num(orb_type_int):
    """
    Convert TurboRVB contracted orbital type number to primitive orbital type number.

    Parameters
    ----------
    orb_type_int : int
        TurboRVB orbital type number (contracted or primitive).

    Returns
    -------
    int
        TurboRVB primitive orbital type number.

    Notes
    -----
    Mapping:

    - 16, 300 -> 16 (s)
    - 36, 400 -> 36 (p)
    - 37, 500 -> 37 (d)
    - 48, 600 -> 48 (f)
    - 51, 700 -> 51 (g)
    - 72, 800 -> 72 (h)
    - 73, 900 -> 73 (i)
    """
    if orb_type_int in {16, 300}:
        return 16
    elif orb_type_int in {36, 400}:
        return 36
    elif orb_type_int in {37, 500}:
        return 37
    elif orb_type_int in {48, 600}:
        return 48
    elif orb_type_int in {51, 700}:
        return 51
    elif orb_type_int in {72, 800}:
        return 72
    elif orb_type_int in {73, 900}:
        return 73
    else:
        logger.error(f"orb_type_int={orb_type_int} is not implemented.")
        raise NotImplementedError(f"Not implemented orb_type_chr={orb_type_int}")


def return_ang_mom(orb_typ_chr):
    """
    Return angular momentum quantum number from orbital type character.

    Parameters
    ----------
    orb_typ_chr : str
        Orbital type character (e.g., "s", "p", "d", "f", "g", "h", "i").

    Returns
    -------
    int
        Angular momentum quantum number (0 for s, 1 for p, etc.).

    Raises
    ------
    NotImplementedError
        If the orbital type character is not implemented.
    """
    if orb_typ_chr in {"s", "s100", "s131"}:
        return 0
    elif orb_typ_chr in {"p", "p103", "p150"}:
        return 1
    elif orb_typ_chr in {"d"}:
        return 2
    elif orb_typ_chr in {"f"}:
        return 3
    elif orb_typ_chr in {"g"}:
        return 4
    elif orb_typ_chr in {"h"}:
        return 5
    elif orb_typ_chr in {"i"}:
        return 6
    else:
        logger.error(f"orb_type={orb_typ_chr} is not implemented.")
        raise NotImplementedError(f"orb_type={orb_typ_chr} is not implemented.")


def return_orbchr(ang_mom):
    """
    Return orbital type character from angular momentum quantum number.

    Parameters
    ----------
    ang_mom : int
        Angular momentum quantum number (0-6).

    Returns
    -------
    str
        Orbital type character ("s", "p", "d", "f", "g", "h", "i").

    Raises
    ------
    NotImplementedError
        If the angular momentum is not in the range 0-6.
    """
    if ang_mom == 0:
        return "s"
    elif ang_mom == 1:
        return "p"
    elif ang_mom == 2:
        return "d"
    elif ang_mom == 3:
        return "f"
    elif ang_mom == 4:
        return "g"
    elif ang_mom == 5:
        return "h"
    elif ang_mom == 6:
        return "i"
    else:
        raise NotImplementedError(f"angmom={ang_mom} is not implemented.")


def return_orb_type_chr(num_orb_type):
    """
    Return orbital type character from TurboRVB orbital type number.

    Parameters
    ----------
    num_orb_type : int
        TurboRVB orbital type number.

    Returns
    -------
    str
        Orbital type character (e.g., "s", "p", "d", "f", "g", "h", "i",
        or special types like "s100", "s131", "p103", "p150").

    Raises
    ------
    NotImplementedError
        If the orbital type number is not implemented.
    """
    if num_orb_type in {16, 300}:
        return "s"
    elif num_orb_type in {100}:
        return "s100"
    elif num_orb_type in {131}:
        return "s131"
    elif num_orb_type in {36, 400}:
        return "p"
    elif num_orb_type in {103}:
        return "p103"
    elif num_orb_type in {150}:
        return "p150"
    elif num_orb_type in {37, 68, 500}:
        return "d"
    elif num_orb_type in {48, 600}:
        return "f"
    elif num_orb_type in {51, 700}:
        return "g"
    elif num_orb_type in {72, 800}:
        return "h"
    elif num_orb_type in {73, 900}:
        return "i"
    elif num_orb_type in {900000}:
        return "hyb"
    elif num_orb_type in {1000000}:
        return "mol"
    elif num_orb_type in {200}:
        return "jas_const"
    else:
        raise NotImplementedError(f"Not suppported orb_type={num_orb_type}")


def return_contraction_flag(orb_type_num):
    if orb_type_num in {16, 100, 131, 36, 103, 150, 37, 68, 48, 51, 72, 73}:
        contraction = False
    elif orb_type_num in {300, 400, 500, 600, 700, 800, 900}:
        contraction = True
    else:
        raise NotImplementedError(f"Not suported orb_type_num={orb_type_num}")
    return contraction


def return_num_twobody_and_flag_onebody(jastrow_type):
    if jastrow_type in {-15, -22}:
        num_twobody = 1
        flag_onebody = True
    elif jastrow_type in {-26, -27}:
        num_twobody = 2
        flag_onebody = True
    elif jastrow_type in {-5, -6}:
        num_twobody = 1
        flag_onebody = False
    elif jastrow_type in {0}:
        num_twobody = 0
        flag_onebody = False
    else:
        raise NotImplementedError(f"Jastrow = {jastrow_type} is not implemented.")

    return num_twobody, flag_onebody


# workarounds!! They work well but not pythonic. to be reafactored?
def pygrep_lineno(file, keyword):
    cmd = f"grep '{keyword}' -n {file} | cut -d ':' -f 1"
    sys_env = os.environ.copy()
    try:
        lineno = int(subprocess.check_output(cmd, shell=True, env=sys_env)) - 1
    except ValueError:
        cmd = f"grep -c '' {file}"
        lineno = int(subprocess.check_output(cmd, shell=True, env=sys_env))

    return lineno


def pysed_replace(file, value, lineno, index, inplace=False):
    if platform.system() == "Darwin":
        if shutil.which("gsed") is None:
            logger.error("The BSD sed on MacOS is not supported.")
            logger.error("Pls. install gsed via homebrew, i.e., brew install gnu-sed")
            raise NotImplementedError
        else:
            sed = "gsed"
            awk = "awk"
    else:
        sed = "sed"
        awk = "awk"

    sys_env = os.environ.copy()
    cmds = []

    lineno += 1

    cmds = [
        f"line=`{sed} -n {lineno}p {file}`",
        f'mod_line=`echo $line |  \
            {awk} \'{{FS=" ";OFS=" "}}{{${index+1}={value}}}1\'`',
    ]

    if inplace:
        cmds += [f"cp {file} {file}_bak"]

    cmds += [
        f'{sed} -i "{lineno}d" {file}',
        f'{sed} -i "{lineno}i \\ $mod_line" {file}',
    ]

    cmd = "; ".join(cmds)
    logger.debug(cmd)
    subprocess.check_call(cmd, shell=True, env=sys_env)


def pysed_replace_lines(
    file, lineno_list, value_list, index_list, inplace=False, cmd_chunk_num=10
):
    if platform.system() == "Darwin":
        if shutil.which("gsed") is None:
            logger.error("The BSD sed on MacOS is not supported.")
            logger.error("Pls. install gsed via homebrew, i.e., brew install gnu-sed")
            raise NotImplementedError
        else:
            sed = "gsed"
            awk = "awk"
    else:
        sed = "sed"
        awk = "awk"

    sys_env = os.environ.copy()

    assert len(lineno_list) == len(value_list)
    assert len(value_list) == len(index_list)

    if inplace:
        cmd = f"cp {file} {file}_bak"
        subprocess.check_call(cmd, shell=True, env=sys_env)

    counter = 0
    cmds = []
    for value_list_l, index_list_l, lineno in zip(value_list, index_list, lineno_list):
        lineno += 1  # because, grep starts from 1, but python starts from 0

        cmds += [f"line=`{sed} -n {lineno}p {file}`"]

        """ old!! Too slow.
        for value, index in zip(value_list_l, index_list_l):
            cmds += [
                f'line=`echo $line |  \
                    {awk} \'{{FS=" ";OFS=" "}}{{${index+1}={value}}}1\'`'
            ]
        """

        # """ new!!
        replaced = ""
        for value, index in zip(value_list_l, index_list_l):
            replaced += f"${index+1}={value};"
            # index+1 # because, grep starts from 1,
            # but python starts from 0
        cmds += [f'line=`echo $line |  {awk} \'{{FS=" ";OFS=" "}}{{{replaced}}}1\'`']
        # """

        # without IOs -> ToDo avoid repating sed!!
        cmds += [
            f'{sed} -i "{lineno}d" {file}',
            f'{sed} -i "{lineno}i \\ $line" {file}',
        ]

        counter += 1
        if counter >= cmd_chunk_num:
            cmd = "; ".join(cmds)
            # logger.debug(cmd)
            subprocess.check_call(cmd, shell=True, env=sys_env)
            counter = 0
            cmds = []

    if len(cmds) != 0:
        cmd = "; ".join(cmds)
        # logger.debug(cmd)
        subprocess.check_call(cmd, shell=True, env=sys_env)
        counter = 0
        cmds = []


def pygetline(
    filename, lineno, clearcache=True
):  # clearchache should be true!! as a default. # reasons for bugs.
    # logger.debug("get line!")
    line = linecache.getline(filename=filename, lineno=lineno + 1)
    if clearcache:
        linecache.clearcache()
    return line


def remove_new_parameter_lines_in_fort10():
    output_buffer = [
        "#!/bin/bash",
        "line_num_key=`grep 'new parameters' -n fort.10 | \
            cut -d ':' -f 1 | head -n 1`",
        'if [ -n "$line_num_key" ]; then',
        "line_num=`expr ${line_num_key} - 1`",
        "head -n ${line_num} fort.10 > fort.10_",
        "mv fort.10_ fort.10",
        "fi",
    ]
    output_buffer = "\n".join(output_buffer)
    with open("./run_local.sh", "w") as f:
        f.writelines(output_buffer)

    sys_env = os.environ.copy()
    cmds = ["chmod +x ./run_local.sh", "./run_local.sh", "./run_local.sh"]
    for cmd in cmds:
        subprocess.check_call(cmd, shell=True, env=sys_env)


if __name__ == "__main__":
    logger = getLogger("pyturbo")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    os.chdir(os.path.join(pyturbo_root, "tests", "fort10"))
    # pysed_replace(file="fort.10", value=5, lineno=2, index=1, inplace=True)
    # value=Value(value=15, lineno=1, index=0, file="fort.10")
    # print(value.v)
    # value.replace(38)
    # print(pygrep_lineno(file="fort.10", keyword="Ion coordinates"))
