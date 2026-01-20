#!python
# -*- coding: utf-8 -*-

"""

convertpfaff genius related classes and methods


"""

# python modules
import os
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbogenius modules
from turbogenius.pyturbo.convertpfaff import Convertpfaff
from turbogenius.utils_workflows.env import turbo_genius_root
from turbogenius.geniusIO import GeniusIO

logger = getLogger("Turbo-Genius").getChild(__name__)


class Convertpfaff_genius(GeniusIO):
    """
    Wrapper class for pyturbo Convertpfaff functionality.

    This class provides a high-level interface to convert wavefunctions
    to Pfaffian form, which is useful for certain types of quantum Monte
    Carlo calculations.

    Parameters
    ----------
    in_fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10_in".
    out_fort10 : str, optional
        Output fort.10 wavefunction file (template), by default "fort.10_out".

    Attributes
    ----------
    in_fort10 : str
        Input fort.10 wavefunction file.
    out_fort10 : str
        Output fort.10 wavefunction file.
    convertpfaff : Convertpfaff
        Underlying pyturbo Convertpfaff instance.
    """

    def __init__(
        self,
        in_fort10: str = "fort.10_in",
        out_fort10: str = "fort.10_out",
    ):
        self.in_fort10 = in_fort10
        self.out_fort10 = out_fort10

        self.convertpfaff = Convertpfaff.parse_from_default_namelist(
            in_fort10=in_fort10, out_fort10=out_fort10
        )

    def generate_input(self):
        """
        Generate input file.

        Notes
        -----
        This method is a placeholder and does nothing for Convertpfaff,
        as it does not require a separate input file generation step.
        """
        pass

    def run_all(
        self,
        rotate_flag: bool = False,
        rotate_angle: float = 0,
        scale_mean_field: int = 1000,
        output_name: str = "out_pfaff",
    ) -> None:
        """
        Generate input files and run the convertpfaff command.

        Parameters
        ----------
        rotate_flag : bool, optional
            If True, rotate the angle of spins, by default False.
        rotate_angle : float, optional
            Rotation angle in units of pi radians, by default 0.
        scale_mean_field : int, optional
            Scaling factor for mean field (used for unpaired cases),
            by default 1000.
        output_name : str, optional
            Output file name, by default "out_pfaff".

        Notes
        -----
        This method calls generate_input() followed by run() to complete
        the conversion process.
        """
        self.generate_input()
        self.run(
            rotate_flag=rotate_flag,
            rotate_angle=rotate_angle,
            scale_mean_field=scale_mean_field,
            output_name=output_name,
        )

    def run(
        self,
        rotate_flag: float = False,
        rotate_angle: float = 0,
        scale_mean_field: int = 1000,
        output_name: str = "out_pfaff",
    ) -> None:
        """
        Run the convertpfaff program.

        Parameters
        ----------
        rotate_flag : bool, optional
            If True, rotate the angle of spins, by default False.
        rotate_angle : float, optional
            Rotation angle in units of pi radians, by default 0.
        scale_mean_field : int, optional
            Scaling factor for mean field (used for unpaired cases),
            by default 1000.
        output_name : str, optional
            Output file name, by default "out_pfaff".

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully (check_results
            indicates failure for any output file).

        Notes
        -----
        This method executes the convertpfaff program and then checks the results.
        An AssertionError is raised if any output file indicates failure.
        """
        self.convertpfaff.run(
            rotate_flag=rotate_flag,
            rotate_angle=rotate_angle,
            scale_mean_field=scale_mean_field,
            output_name=output_name,
        )
        flags = self.convertpfaff.check_results(output_names=[output_name])
        assert all(flags)

    def check_results(self, output_names: Optional[list] = None) -> bool:
        """
        Check the results of the convertpfaff program execution.

        Parameters
        ----------
        output_names : list, optional
            List of output file names to check. If None, defaults to
            ["out_pfaff"], by default None.

        Returns
        -------
        bool
            True if all the runs were successful, False if an error is
            detected in the output files.

        Notes
        -----
        This method checks the output files for successful completion.
        Currently, this is a placeholder and always returns True.
        """
        if output_names is None:
            output_names = ["out_pfaff"]
        return self.convertpfaff.check_results(output_names=output_names)


if __name__ == "__main__":
    from turbogenius.convertfort10_genius import Convertfort10_genius

    logger = getLogger("Turbo-Genius")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    os.chdir(os.path.join(turbo_genius_root, "tests", "convertfort10"))

    # moved to examples
    convertfort10_genius = Convertfort10_genius()
    convertfort10_genius.generate_input()
    convertfort10_genius.run()
