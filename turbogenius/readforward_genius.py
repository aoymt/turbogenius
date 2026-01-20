#!python
# -*- coding: utf-8 -*-

"""

Readforward related classes and methods

Todo:
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.

"""

# python modules
import os
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbogenius modules
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.pyturbo.readforward import Readforward
from turbogenius.utils_workflows.env import turbo_genius_root
from turbogenius.geniusIO import GeniusIO

logger = getLogger("Turbo-Genius").getChild(__name__)


class Readforward_genius(GeniusIO):
    """
    Wrapper class for pyturbo readforward functionality.

    This class provides a high-level interface to the readforward program,
    which is used for analyzing correlation functions and performing
    correlated sampling calculations.

    Parameters
    ----------
    in_fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10_in".
    corr_fort10 : str, optional
        Reference fort.10 wavefunction file for correlated sampling,
        by default "fort.10_corr".
    bin_block : int, optional
        Binning length for correlation function analysis, by default 10.
    warmupblocks : int, optional
        Number of disregarded blocks at the beginning, by default 2.
    corr_sampling : bool, optional
        If True, enable correlated sampling, by default True.

    Attributes
    ----------
    in_fort10 : str
        Input fort.10 wavefunction file.
    corr_fort10 : str
        Reference fort.10 wavefunction file for correlated sampling.
    readforward : Readforward
        Underlying pyturbo Readforward instance.
    io_fort10 : IO_fort10
        IO_fort10 instance for reading fort.10 files.
    """

    def __init__(
        self,
        in_fort10: str = "fort.10_in",
        corr_fort10: str = "fort.10_corr",
        bin_block: int = 10,
        warmupblocks: int = 2,
        corr_sampling: bool = True,
    ):

        self.in_fort10 = in_fort10
        self.corr_fort10 = corr_fort10

        self.readforward = Readforward.parse_from_default_namelist(
            in_fort10=in_fort10
        )
        self.io_fort10 = IO_fort10(self.in_fort10)

        self.readforward.set_parameter(
            parameter="bin_length", value=bin_block, namelist="&corrfun"
        )
        self.readforward.set_parameter(
            parameter="initial_bin", value=warmupblocks, namelist="&corrfun"
        )
        if corr_sampling:
            self.readforward.set_parameter(
                parameter="correlated_samp",
                value=".true.",
                namelist="&corrfun",
            )
        else:
            pass
            # self.convertfort10.comment_out(parameter="nz")

    def run_all(
        self,
        input_name: str = "readforward.input",
        output_name: str = "out_readforward",
    ) -> None:
        """
        Generate input files and run the command.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "readforward.input".
        output_name : str, optional
            Output file name, by default "out_readforward".
        """
        self.generate_input(input_name=input_name)
        self.run(input_name=input_name, output_name=output_name)

    def generate_input(self, input_name: str = "readforward.input") -> None:
        """
        Generate input file.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "readforward.input".
        """
        self.readforward.generate_input(input_name=input_name)

    def run(
        self,
        input_name: str = "readforward.input",
        output_name: str = "out_readforward",
    ) -> None:
        """
        Run the readforward program.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "readforward.input".
        output_name : str, optional
            Output file name, by default "out_readforward".

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully (check_results
            indicates failure for any output file).

        Notes
        -----
        This method executes the readforward program and then checks the results.
        An AssertionError is raised if any output file indicates failure.
        """
        self.readforward.run(input_name=input_name, output_name=output_name)
        flags = self.readforward.check_results(output_names=[output_name])
        assert all(flags)

    def check_results(self, output_names: Optional[list] = None) -> bool:
        """
        Check the result.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names to check. If None, defaults to
            ["out_readforward"], by default None.

        Returns
        -------
        bool
            True if all the runs were successful, False if an error is
            detected in the files.
        """
        if output_names is None:
            output_names = ["out_readforward"]
        return self.readforward.check_results(output_names=output_names)


if __name__ == "__main__":
    from turbogenius.convertfort10_genius import Convertfort10_genius

    logger = getLogger("Turbo-Genius")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter(
        "%(name)s - %(levelname)s - %(lineno)d - %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    os.chdir(os.path.join(turbo_genius_root, "tests", "convertfort10"))

    # moved to examples
    convertfort10_genius = Convertfort10_genius()
    convertfort10_genius.generate_input()
    convertfort10_genius.run()
