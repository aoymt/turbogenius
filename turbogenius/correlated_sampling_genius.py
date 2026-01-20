#!python
# -*- coding: utf-8 -*-

"""

Correlated sampling related classes and methods

Todo:
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.

"""

# python modules
import os
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbogenius modules
from turbogenius.pyturbo.vmc import VMC
from turbogenius.pyturbo.readforward import Readforward
from turbogenius.utils_workflows.env import turbo_genius_root
from turbogenius.geniusIO import GeniusIO

logger = getLogger("Turbo-Genius").getChild(__name__)


class Correlated_sampling_genius(GeniusIO):
    """
    Wrapper class for correlated sampling calculations.

    This class provides a high-level interface for performing correlated
    sampling calculations using VMC and readforward programs. It is useful
    for comparing wavefunctions and computing energy differences.

    Parameters
    ----------
    in_fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10_in".
    corr_fort10 : str, optional
        Reference fort.10 wavefunction file for correlated sampling,
        by default "fort.10_corr".
    vmcsteps : int, optional
        Total number of MCMC steps, by default 100.
    bin_block : int, optional
        Binning length for correlation function analysis, by default 10.
    warmupblocks : int, optional
        Number of disregarded blocks at the beginning, by default 2.
    num_walkers : int, optional
        Number of walkers. If -1, uses the number of MPI processes,
        by default -1.
    maxtime : int, optional
        Maximum time in seconds, by default 172800.
    twist_average : bool, optional
        Twist average flag, True or False, by default False.
    kpoints : list, optional
        k Monkhorst-Pack grids, [kx,ky,kz,nx,ny,nz], where kx,y,z are grids
        and nx,y,z are shift (0) or no shift (1), by default [1, 1, 1, 0, 0, 0].

    Attributes
    ----------
    in_fort10 : str
        Input fort.10 wavefunction file.
    corr_fort10 : str
        Reference fort.10 wavefunction file.
    vmc : VMC
        Underlying pyturbo VMC instance.
    readforward : Readforward
        Underlying pyturbo Readforward instance.
    """

    def __init__(
        self,
        in_fort10: str = "fort.10_in",
        corr_fort10: str = "fort.10_corr",
        vmcsteps: int = 100,
        bin_block: int = 10,
        warmupblocks: int = 2,
        num_walkers: int = -1,  # default -1 -> num of MPI process.
        maxtime: int = 172800,
        twist_average: bool = False,
        kpoints: Optional[list] = None,
    ):
        if kpoints is None:
            kpoints = [1, 1, 1, 0, 0, 0]

        self.in_fort10 = in_fort10
        self.corr_fort10 = corr_fort10

        self.vmcsteps = vmcsteps
        self.bin_block = bin_block
        self.warmupblocks = warmupblocks
        self.num_walkers = num_walkers
        self.maxtime = maxtime
        self.twist_average = twist_average
        self.kpoints = kpoints

        # VMC
        self.vmc = VMC.parse_from_default_namelist(
            in_fort10=in_fort10, twist_average=twist_average
        )
        if vmcsteps < 40 * bin_block + bin_block * warmupblocks:
            logger.warning(
                f"vmcsteps = {vmcsteps} is too small! < 40 * bin_block + bin_block * warmupblocks = {40 * bin_block + bin_block * warmupblocks}"
            )
            logger.warning(
                f"vmcsteps = {vmcsteps} is set to 40 * bin_block + bin_block * warmupblocks = {40 * bin_block + bin_block * warmupblocks}"
            )
            vmcsteps = 40 * bin_block + bin_block * warmupblocks
        self.vmc.set_parameter(
            parameter="ngen", value=vmcsteps, namelist="&simulation"
        )
        self.vmc.set_parameter(
            parameter="maxtime", value=maxtime, namelist="&simulation"
        )
        if num_walkers != -1:
            self.vmc.set_parameter(
                parameter="nw", value=num_walkers, namelist="&simulation"
            )

        self.vmc.set_parameter(parameter="iread", value=3, namelist="&readio")

        # kpoints
        if self.twist_average:  # not 0 (= not False)!!
            if self.twist_average == 1:  # True case, Monkhorst-Pack algorithm
                assert len(self.kpoints) == 6
                nkx, nky, nkz, kx, ky, kz = self.kpoints
                self.vmc.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                self.vmc.set_parameter(
                    parameter="yeswrite10",
                    value=".true.",
                    namelist="&optimization",
                )
                self.vmc.set_parameter(
                    parameter="kp_type", value=1, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="nkx", value=nkx, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="nky", value=nky, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="nkz", value=nkz, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="kx", value=kx, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="ky", value=ky, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="kz", value=kz, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="skip_equivalence",
                    value=".true.",
                    namelist="&kpoints",
                )
                self.vmc.set_parameter(
                    parameter="double_kpgrid",
                    value=".true.",
                    namelist="&kpoints",
                )
            elif self.twist_average == 2:  # k-points are set from the user
                assert len(self.kpoints) == 2
                kpoints_up, kpoints_dn = self.kpoints
                assert len(kpoints_up) == len(kpoints_dn)
                for kup, kdn in zip(kpoints_up, kpoints_dn):
                    assert len(kup) == 4  # kx, ky, kz, wkp for up
                    assert len(kdn) == 4  # kx, ky, kz, wkp for dn
                self.vmc.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                self.vmc.set_parameter(
                    parameter="yeswrite10",
                    value=".true.",
                    namelist="&optimization",
                )
                self.vmc.set_parameter(
                    parameter="kp_type", value=2, namelist="&kpoints"
                )
                self.vmc.set_parameter(
                    parameter="nk1",
                    value=len(kpoints_up) + len(kpoints_dn),
                    namelist="&kpoints",
                )
                self.vmc.manual_kpoints = self.kpoints

            else:
                logger.error(
                    f"twist_average = {self.twist_average} is not implemented."
                )
                raise NotImplementedError

        # readforward
        self.readforward = Readforward.parse_from_default_namelist(
            in_fort10=in_fort10
        )
        self.readforward.set_parameter(
            parameter="bin_length", value=bin_block, namelist="&corrfun"
        )
        self.readforward.set_parameter(
            parameter="initial_bin", value=warmupblocks, namelist="&corrfun"
        )
        self.readforward.set_parameter(
            parameter="correlated_samp", value=".true.", namelist="&corrfun"
        )

    def run_all(
        self,
        input_name: str = "datasvmc.input",
        vmc_output_name: str = "out_vmc",
        readforward_output_name: str = "out_readforward",
    ) -> None:
        """
        Generate input files and run the correlated sampling calculations.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "datasvmc.input".
        vmc_output_name : str, optional
            VMC output file name, by default "out_vmc".
        readforward_output_name : str, optional
            Readforward output file name, by default "out_readforward".

        Notes
        -----
        This method runs both VMC and readforward programs sequentially
        to perform correlated sampling calculations.
        """
        self.vmc.generate_input(input_name=input_name)
        self.vmc.run(input_name=input_name, output_name=vmc_output_name)
        self.readforward.generate_input(input_name=input_name)
        self.readforward.run(
            input_name=input_name, output_name=readforward_output_name
        )

    def generate_input(self, input_name: str = "datasvmc.input") -> None:
        """
        Generate input files for VMC and readforward programs.

        Parameters
        ----------
        input_name : str, optional
            Input file name for VMC, by default "datasvmc.input".

        Notes
        -----
        This method generates both datasvmc.input (for VMC) and
        readforward.input (for readforward) files.
        """
        self.vmc.generate_input(input_name=input_name)
        self.readforward.generate_input(input_name="readforward.input")

    def run(
        self,
        input_name: str = "datasvmc.input",
        vmc_output_name: str = "out_vmc",
        readforward_output_name: str = "out_readforward",
    ) -> None:
        """
        Run the correlated sampling calculations.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "datasvmc.input".
        vmc_output_name : str, optional
            VMC output file name, by default "out_vmc".
        readforward_output_name : str, optional
            Readforward output file name, by default "out_readforward".

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully (check_results
            indicates failure for any output file).

        Notes
        -----
        This method runs VMC first, then readforward. Both results are checked,
        and an AssertionError is raised if either calculation fails.
        """
        self.vmc.run(input_name=input_name, output_name=vmc_output_name)
        flags = self.vmc.check_results(output_names=[vmc_output_name])
        assert all(flags)
        self.readforward.run(
            input_name=input_name, output_name=readforward_output_name
        )
        flags = self.readforward.check_results(
            output_names=[readforward_output_name]
        )
        assert all(flags)

    def check_results(
        self,
        vmc_output_names: Optional[list] = None,
        readforward_output_names: Optional[list] = None,
    ) -> bool:
        """
        Check the results of the correlated sampling calculations.

        Parameters
        ----------
        vmc_output_names : list, optional
            List of VMC output file names to check. If None, defaults to
            ["out_vmc"], by default None.
        readforward_output_names : list, optional
            List of readforward output file names to check. If None,
            defaults to ["out_readforward"], by default None.

        Returns
        -------
        bool
            True if all the runs were successful, False if an error is
            detected in the output files.

        Notes
        -----
        This method checks both VMC and readforward output files for
        successful completion.
        """
        if vmc_output_names is None:
            vmc_output_names = ["out_vmc"]
        if readforward_output_names is None:
            readforward_output_names = ["out_readforward"]
        return self.readforward.check_results(
            output_names=readforward_output_names
        ) + self.vmc.check_results(output_names=vmc_output_names)


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
