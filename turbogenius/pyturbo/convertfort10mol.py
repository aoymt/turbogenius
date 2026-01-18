#!python
# -*- coding: utf-8 -*-

"""

pyturbo: convertfort10mol related classes and methods

"""
# python modules
import os
import re
from typing import Optional

# pyturbo modules
from turbogenius.pyturbo.namelist import Namelist
from turbogenius.pyturbo.fortranIO import FortranIO
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.pyturbo.utils.env import turbo_convertfort10mol_run_command
from turbogenius.pyturbo.utils.env import pyturbo_data_dir
from turbogenius.pyturbo.utils.utility import file_check
from turbogenius.pyturbo.utils.execute import run

from logging import getLogger, StreamHandler, Formatter

logger = getLogger("pyturbo").getChild(__name__)


class Convertfort10mol(FortranIO):
    """
    Wrapper class for TurboRVB convertfort10mol.x program.

    This class provides an interface to convert fort.10 wavefunction files
    to molecular orbital format, with options to add random molecular orbitals.

    Parameters
    ----------
    in_fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10_in".
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
        in_fort10: str = "fort.10_in",
        namelist: Optional[Namelist] = None,
    ):
        """
        Initialize the Convertfort10mol class.

        Parameters
        ----------
        in_fort10 : str, optional
            Input fort.10 wavefunction file, by default "fort.10_in".
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
        self.in_fort10 = in_fort10
        self.namelist = namelist

        file_check(in_fort10)
        if IO_fort10(fort10=in_fort10).pp_flag:
            file_check("pseudo.dat")

    def __str__(self) -> str:
        """
        Return string representation of the Convertfort10mol object.

        Returns
        -------
        str
            String description of the object.
        """
        return str(f"{self.__class__.__name__} class")

    def sanity_check(self) -> None:
        """
        Perform sanity checks on the input parameters.

        Notes
        -----
        This method is a placeholder and does nothing. It should be
        implemented to validate input parameters.
        """
        pass

    def generate_input(self, input_name: str = "convertfort10mol.input") -> None:
        """
        Generate input file for the convertfort10mol program.

        Parameters
        ----------
        input_name : str, optional
            Output input file name, by default "convertfort10mol.input".
        """
        self.namelist.write(input_name)
        logger.info(f"{input_name} has been generated.")

    def run(
        self,
        input_name: str = "convertfort10mol.input",
        output_name: str = "out_mol",
    ) -> None:
        """
        Run the convertfort10mol program.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "convertfort10mol.input".
        output_name : str, optional
            Output file name, by default "out_mol".

        Raises
        ------
        subprocess.CalledProcessError
            If the program execution fails.
        """
        run(
            turbo_convertfort10mol_run_command,
            input_name=input_name,
            output_name=output_name,
        )

    def check_results(self, output_names: Optional[list] = None) -> list:
        """
        Check the results of the convertfort10mol program execution.

        Parameters
        ----------
        output_names : list, optional
            List of output file names to check. If None, defaults to
            ["out_mol"], by default None.

        Returns
        -------
        list of bool
            List of boolean flags indicating success for each output file.
            True if the file contains "Time.*change.*fort\.10", False otherwise.
        """
        if output_names is None:
            output_names = ["out_mol"]
        flags = []
        for output_name in output_names:
            file_check(output_name)
            with open(output_name, "r") as f:
                lines = f.readlines()
            if any([re.match(r".*Time.*change.*fort\.10.*", line) for line in lines]):
                flags.append(True)
            else:
                flags.append(False)
        return flags

    @staticmethod
    def read_default_namelist(in_fort10: str = "fort.10_in"):
        """
        Read default namelist values from the turbogenius database.

        Parameters
        ----------
        in_fort10 : str, optional
            Input fort.10 file. The number of molecular orbitals (nmol) is
            automatically set based on this file, by default "fort.10_in".

        Returns
        -------
        Namelist
            Namelist object with default parameter values. The "nmol" parameter
            in the "&molec_info" namelist is set to nel/2 from the fort.10 file.

        Raises
        ------
        FileNotFoundError
            If the fort.10 file is not found.
        """
        convertfort10mol_default_file = os.path.join(
            pyturbo_data_dir, "convertfort10mol", "convertfort10mol.input"
        )
        namelist = Namelist.parse_namelist_from_file(convertfort10mol_default_file)
        io_fort10 = IO_fort10(in_fort10)
        nmol = int(io_fort10.f10header.nel / 2)
        namelist.set_parameter("nmol", nmol, "&molec_info")
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
    def parse_from_default_namelist(cls, in_fort10: str = "fort.10_in"):
        """
        Create a Convertfort10mol instance with default namelist values.

        Parameters
        ----------
        in_fort10 : str, optional
            Input fort.10 wavefunction file, by default "fort.10_in".

        Returns
        -------
        Convertfort10mol
            Convertfort10mol instance with default namelist values.
        """
        namelist = cls.read_default_namelist(in_fort10=in_fort10)
        return cls(in_fort10=in_fort10, namelist=namelist)

    @classmethod
    def parse_from_file(cls, file: str, in_fort10: str = "fort.10_in"):
        """
        Create a Convertfort10mol instance from a namelist file.

        Parameters
        ----------
        file : str
            Path to the input file.
        in_fort10 : str, optional
            Input fort.10 wavefunction file, by default "fort.10_in".

        Returns
        -------
        Convertfort10mol
            Convertfort10mol instance with namelist values from the file.
        """
        namelist = Namelist.parse_namelist_from_file(file)
        return cls(in_fort10=in_fort10, namelist=namelist)


if __name__ == "__main__":
    logger = getLogger("pyturbo")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # moved to examples.
