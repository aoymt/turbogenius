#!python
# -*- coding: utf-8 -*-

"""

pyturbo: readforward related classes and methods

Todo:
    * docstrings are not completed.
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.
    * implementing __str__ method.
    * implementing sanity_check method.

"""

# python modules
import os
import re
from typing import Optional

# pyturbo modules
from turbogenius.pyturbo.namelist import Namelist
from turbogenius.pyturbo.fortranIO import FortranIO
from turbogenius.pyturbo.utils.env import turbo_readforward_run_command
from turbogenius.pyturbo.utils.env import pyturbo_data_dir, pyturbo_root
from turbogenius.pyturbo.utils.utility import file_check
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.pyturbo.utils.execute import run


from logging import getLogger, StreamHandler, Formatter

logger = getLogger("pyturbo").getChild(__name__)


class Readforward(FortranIO):
    """
    Wrapper class for TurboRVB readforward program.

    This class provides an interface to the readforward program, which is
    used for analyzing correlation functions and performing correlated
    sampling calculations.

    Parameters
    ----------
    in_fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10".
    namelist : Namelist, optional
        Namelist object containing program parameters. If None, an empty
        Namelist is created, by default None.

    Attributes
    ----------
    in_fort10 : str
        Input fort.10 wavefunction file.
    namelist : Namelist
        Namelist object containing program parameters.
    """

    def __init__(
        self,
        in_fort10: str = "fort.10",
        namelist: Optional[Namelist] = None,
    ):
        """
        Initialize the Readforward class.

        Parameters
        ----------
        in_fort10 : str, optional
            Input fort.10 wavefunction file, by default "fort.10".
        namelist : Namelist, optional
            Namelist object containing program parameters. If None, an empty
            Namelist is created, by default None.

        Raises
        ------
        FileNotFoundError
            If the fort.10 file or pseudo.dat (if needed) is not found.
        """
        if namelist is None:
            namelist = Namelist()

        """
        input values
        """
        file_check(in_fort10)
        if IO_fort10(fort10=in_fort10).pp_flag:
            file_check("pseudo.dat")
        self.in_fort10 = in_fort10
        self.namelist = namelist

    def __str__(self):
        """
        Return string representation of the Readforward object.

        Returns
        -------
        str
            String description of the object.
        """
        output = [
            "TurboRVB readforward python wrapper",
        ]
        return "\n".join(output)

    def sanity_check(self):
        """
        Perform sanity checks on the input parameters.

        Notes
        -----
        This method is a placeholder and does nothing. It should be
        implemented to validate input parameters.
        """
        pass

    def generate_input(self, input_name: str = "readforward.input"):
        """
        Generate input file for the readforward program.

        Parameters
        ----------
        input_name : str, optional
            Output input file name, by default "readforward.input".
        """
        self.namelist.write(input_name)
        logger.info(f"{input_name} has been generated.")

    def run(
        self,
        input_name: str = "datasvmc.input",
        output_name: str = "out_readforward",
    ):
        """
        Run the readforward program.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "datasvmc.input".
        output_name : str, optional
            Output file name, by default "out_readforward".

        Raises
        ------
        subprocess.CalledProcessError
            If the program execution fails.
        """
        run(
            turbo_readforward_run_command,
            input_name=input_name,
            output_name=output_name,
        )

    def check_results(self, output_names: Optional[list] = None):
        """
        Check the results of the readforward program execution.

        Parameters
        ----------
        output_names : list, optional
            List of output file names to check. If None, defaults to
            ["out_readforward"], by default None.

        Returns
        -------
        list of bool
            List of boolean flags indicating success for each output file.
            True if the file contains "total.*#.*bin.*considered", False otherwise.
        """
        if output_names is None:
            output_names = ["out_readforward"]
        flags = []
        for output_name in output_names:
            file_check(output_name)
            with open(output_name, "r") as f:
                lines = f.readlines()
            if any(
                [
                    re.match(r".*total.*#.*bin.*considered.*", line)
                    for line in lines
                ]
            ):
                flags.append(True)
            else:
                flags.append(False)
        return flags

    @staticmethod
    def read_default_namelist(in_fort10: str = "fort.10"):
        """
        Read default namelist values from the turbogenius database.

        Parameters
        ----------
        in_fort10 : str, optional
            Input fort.10 file (currently not used), by default "fort.10".

        Returns
        -------
        Namelist
            Namelist object with default parameter values.
        """
        readforward_default_file = os.path.join(
            pyturbo_data_dir, "readforward", "readforward.input"
        )
        namelist = Namelist.parse_namelist_from_file(readforward_default_file)
        return namelist

    @staticmethod
    def read_namelist_from_file(file: str):
        """
        Read namelist values from a specified file.

        Parameters
        ----------
        file : str
            Path to the input file.

        Returns
        -------
        Namelist
            Namelist object with parameter values from the file.
        """
        namelist = Namelist.parse_namelist_from_file(file)
        return namelist

    @classmethod
    def parse_from_default_namelist(cls, in_fort10: str = "fort.10"):
        """
        Create a Readforward instance with default namelist values.

        Parameters
        ----------
        in_fort10 : str, optional
            Input fort.10 wavefunction file, by default "fort.10".

        Returns
        -------
        Readforward
            Readforward instance with default namelist values.
        """
        namelist = cls.read_default_namelist(in_fort10=in_fort10)
        return cls(in_fort10=in_fort10, namelist=namelist)

    @classmethod
    def parse_from_file(cls, file, in_fort10: str = "fort.10"):
        """
        Create a Readforward instance from a namelist file.

        Parameters
        ----------
        file : str
            Path to the input file.
        in_fort10 : str, optional
            Input fort.10 wavefunction file, by default "fort.10".

        Returns
        -------
        Readforward
            Readforward instance with namelist values from the file.
        """
        namelist = Namelist.parse_namelist_from_file(file)
        return cls(in_fort10=in_fort10, namelist=namelist)


if __name__ == "__main__":
    logger = getLogger("pyturbo")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter(
        "%(name)s - %(levelname)s - %(lineno)d - %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # moved to examples.

    readforward_test_dir = os.path.join(pyturbo_root, "tests", "readforward")
    os.chdir(readforward_test_dir)

    readforward = Readforward().parse_from_default_namelist()
    readforward.generate_input()
