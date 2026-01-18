#!python -u
# -*- coding: utf-8 -*-

"""

pyturbo: namelist related classes and methods

Todo:
    * docstrings are not completed.
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.
    * implementing __str__ method.
    * implementing sanity_check method.

"""

# python modules
import re
from typing import Optional, Union

# from collections import OrderedDict

# turbo-genius modules
from turbogenius.pyturbo.utils.utility import get_str_variable_type_auto

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("pyturbo").getChild(__name__)


class Namelist:
    """
    Class for managing Fortran namelist parameters.

    This class provides functionality to read, write, and manipulate
    Fortran namelist parameters used in TurboRVB input files.

    Parameters
    ----------
    namelist : dict, optional
        Dictionary containing namelist data. Keys are namelist names (e.g., "&simulation"),
        values are dictionaries of parameter names and values, by default None.

    Attributes
    ----------
    parameters : dict
        Property to access the namelist dictionary.

    Examples
    --------
    >>> namelist = Namelist()
    >>> namelist.set_parameter("ngen", 100, "&simulation")
    >>> namelist.write("input.inp")
    """

    def __init__(self, namelist: dict = None):
        """
        Initialize the Namelist object.

        Parameters
        ----------
        namelist : dict, optional
            Dictionary containing namelist data, by default None.
        """
        if namelist is None:
            namelist = dict()
        self.__namelist = namelist

    @property
    def parameters(self):
        """
        Get the namelist dictionary.

        Returns
        -------
        dict
            Dictionary containing all namelists and their parameters.
        """
        return self.__namelist

    def set_parameter(
        self,
        parameter: str,
        value: Union[int, float, str],
        namelist: Optional[str] = None,
    ):
        """
        Set a parameter value in the namelist.

        Parameters
        ----------
        parameter : str
            Parameter name to set.
        value : int, float, or str
            Parameter value.
        namelist : str, optional
            Namelist name (e.g., "&simulation"). If None, searches all namelists
            for the parameter, by default None.

        Returns
        -------
        bool
            True if the parameter was set successfully.

        Raises
        ------
        KeyError
            If the parameter is not found in the namelist when namelist is None,
            or if the specified namelist does not exist.
        """
        if namelist is None:
            for key, parameters in self.__namelist.items():
                if parameter in parameters.keys():
                    self.__namelist[key][parameter] = value
                    return True
            raise KeyError(
                f"parameter={parameter} is not defined in the defined namelist. Specify a namelist."
            )

        else:  # namelist is not None
            try:
                self.__namelist[namelist][parameter] = value
            except KeyError:
                logger.error(f"{namelist} is not defined in the namelist.")
                raise KeyError

    def get_parameter(self, parameter, namelist=None):
        """
        Get a parameter value from the namelist.

        Parameters
        ----------
        parameter : str
            Parameter name to get.
        namelist : str, optional
            Namelist name. If None, searches all namelists, by default None.

        Returns
        -------
        int, float, or str
            Parameter value.

        Raises
        ------
        ValueError
            If the parameter is not found in the namelist.
        """
        for key, parameters in self.__namelist.items():
            if namelist is not None and key != namelist:
                continue
            if parameter in parameters.keys():
                return parameters[parameter]
        logger.error(f"{parameter} is not defined in the namelist")
        raise ValueError
        # return None

    def get_parameters(self):
        """
        Get all parameters from the namelist.

        Returns
        -------
        dict
            Dictionary containing all namelists and their parameters.
        """
        return self.__namelist

    def comment_out(self, parameter):
        """
        Comment out a parameter in the namelist.

        Parameters
        ----------
        parameter : str
            Parameter name to comment out.

        Notes
        -----
        This method removes the parameter from the active namelist and adds
        it with a "!" prefix to comment it out. The original value is preserved.
        """
        for key, parameters in self.__namelist.items():
            if parameter in parameters.keys():
                value = self.__namelist[key].pop(parameter)
                self.__namelist[key]["!" + parameter] = value

    def write(self, file_name):
        """
        Write the namelist to a file.

        Parameters
        ----------
        file_name : str
            Output file name.

        Notes
        -----
        The file is written in Fortran namelist format with proper formatting.
        String values are enclosed in single quotes, and boolean values are
        written as ".true." or ".false.".
        """
        output = []

        for key, parameters in self.__namelist.items():
            output.append(f"{key}\n")
            namelist_values = parameters
            for key, value in namelist_values.items():
                if (
                    type(value) == str
                    and not re.match(".*true.*", value)
                    and not re.match(".*false.*", value)
                ):
                    output.append(f"    {key}='{value}'\n")
                else:
                    output.append(f"    {key}={value}\n")
            output.append("/\n\n")

        with open(file_name, "w") as f:
            f.writelines(output)

    @staticmethod
    def read_parameters_from_file(file_name):
        """
        Read parameters from a Fortran namelist file.

        Parameters
        ----------
        file_name : str
            Path to the input file.

        Returns
        -------
        dict
            Dictionary containing namelists and their parameters.
            Keys are namelist names (e.g., "&simulation"), values are
            dictionaries of parameter names and values.

        Notes
        -----
        This method parses Fortran namelist format files, skipping comment
        lines (starting with "!") and stopping at "ATOMIC_POSITIONS" if present.
        """
        with open(file_name, "r") as f:
            input_lines = f.readlines()

        namelist = []
        namelist_l = []
        namelist_d = {}
        namelist_d_ordered = dict()
        read_flag = False

        for line in input_lines:
            if re.match("^[\s]*!.*", line):
                continue
            if re.match(".*ATOMIC_POSITIONS.*", line):
                break
            if re.match(".*&.*", line):
                namelist.append(line.replace("\n", ""))
                namelist_d = {}
                read_flag = True
                continue
            if re.match(".*/.*", line):
                namelist_l.append(namelist_d)
                read_flag = False
                continue
            if read_flag:
                # print(line)
                # key,value,_= line.replace("\n","").replace(" ","").split("=")
                key, value, *_ = re.split(
                    "[=,!]", line.replace("\n", "").replace(" ", "")
                )
                converted_value = get_str_variable_type_auto(value)
                namelist_d[key] = converted_value

        for name, parameters in zip(namelist, namelist_l):
            namelist_d_ordered[name] = parameters

        return namelist_d_ordered

    @classmethod
    def parse_namelist_from_file(cls, file_name):
        """
        Create a Namelist instance from a file.

        Parameters
        ----------
        file_name : str
            Path to the input file.

        Returns
        -------
        Namelist
            Namelist instance with parameters read from the file.
        """
        namelist_d_ordered = cls.read_parameters_from_file(file_name)
        return cls(namelist=namelist_d_ordered)


if __name__ == "__main__":
    logger = getLogger("pyturbo")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # moved to examples
