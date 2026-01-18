#!python
# -*- coding: utf-8 -*-

"""

lrdmc genius related classes and methods

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
from turbogenius.pyturbo.lrdmc import LRDMC
from turbogenius.utils_workflows.env import turbo_genius_root
from turbogenius.utils_workflows.utility import get_nonlocalmoves_setting
from turbogenius.geniusIO import GeniusIO
from turbogenius.pyturbo.io_fort10 import IO_fort10

logger = getLogger("Turbo-Genius").getChild(__name__)


class LRDMC_genius(GeniusIO):
    """
    Wrapper class for pyturbo LRDMC (Lattice Regularized Diffusion Monte Carlo) functionality.

    This class provides a high-level interface to perform LRDMC calculations,
    including energy and force calculations with optional twist averaging.

    Parameters
    ----------
    fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10".
    lrdmcsteps : int, optional
        Total number of MCMC steps, by default 100.
    alat : float, optional
        Lattice space in Bohr, by default -0.20.
    time_branching : float, optional
        Interval between two branching steps in atomic units, by default 0.10.
    etry : float, optional
        Trial energy in Hartree, by default 0.0.
    num_walkers : int, optional
        Number of walkers. If -1, uses the number of MPI processes,
        by default -1.
    pw_regularization : float, optional
        Pathak-Wagner regularization parameter. If > 0.0, the regularization
        is turned on, by default 0.00.
    maxtime : int, optional
        Maximum time in seconds, by default 172800.
    twist_average : bool, optional
        Twist average flag, True or False, by default False.
    kpoints : list, optional
        k Monkhorst-Pack grids, [kx,ky,kz,nx,ny,nz], where kx,y,z are grids
        and nx,y,z are shift (0) or no shift (1), by default [1, 1, 1, 0, 0, 0].
    force_calc_flag : bool, optional
        If True, compute energy and force. If False, compute only energy,
        by default False.
    nonlocalmoves : str, optional
        Treatment of locality approximation. Choose from "tmove", "dla", "dlatm",
        by default "dla".

    Attributes
    ----------
    fort10 : str
        Input fort.10 wavefunction file.
    lrdmc : LRDMC
        Underlying pyturbo LRDMC instance.
    io_fort10 : IO_fort10
        IO_fort10 instance for reading fort.10 files.
    energy : float or None
        Energy value (set after calculation).
    energy_error : float or None
        Energy error (set after calculation).
    forces : numpy.ndarray or None
        Forces array (3 * natom matrix, set after calculation).
    forces_error : numpy.ndarray or None
        Forces error array (set after calculation).
    estimated_time_for_1_generation : float or None
        Estimated time for one generation (set after calculation).
    """

    def __init__(
        self,
        fort10: str = "fort.10",
        lrdmcsteps: int = 100,
        alat: float = -0.20,
        time_branching: float = 0.10,
        etry: float = 0.0,
        num_walkers: int = -1,  # default -1 -> num of MPI process.
        pw_regularization: float = 0.00,
        maxtime: int = 172800,
        twist_average: bool = False,
        kpoints: Optional[list] = None,
        force_calc_flag: bool = False,
        nonlocalmoves: str = "dla",  # tmove, dla, dlatm
    ):
        if kpoints is None:
            kpoints = [1, 1, 1, 0, 0, 0]

        self.force_calc_flag = force_calc_flag
        self.twist_average = twist_average
        self.kpoints = kpoints

        self.estimated_time_for_1_generation = None

        self.io_fort10=IO_fort10(fort10)

        self.energy = None
        self.energy_error = None
        self.forces = None  # np.array([[]]) # 3 * natom matrix
        self.forces_error = None  # np.array([[]])  # 3 * natom matrix

        self.lrdmc = LRDMC.parse_from_default_namelist(
            in_fort10=fort10, twist_average=twist_average
        )
        self.lrdmc.set_parameter(
            parameter="ngen", value=lrdmcsteps, namelist="&simulation"
        )
        self.lrdmc.set_parameter(
            parameter="maxtime", value=maxtime, namelist="&simulation"
        )
        if num_walkers != -1:
            self.lrdmc.set_parameter(
                parameter="nw", value=num_walkers, namelist="&simulation"
            )

        self.lrdmc.set_parameter(parameter="etry", value=etry, namelist="&dmclrdmc")
        self.lrdmc.set_parameter(parameter="alat", value=alat, namelist="&dmclrdmc")
        self.lrdmc.set_parameter(
            parameter="tbra", value=time_branching, namelist="&dmclrdmc"
        )
        io_fort10 = IO_fort10(fort10=fort10)
        if io_fort10.pp_flag:
            typereg, npow = get_nonlocalmoves_setting(nonlocalmoves=nonlocalmoves)
            self.lrdmc.set_parameter(
                parameter="typereg", value=typereg, namelist="&dmclrdmc"
            )
            self.lrdmc.set_parameter(parameter="npow", value=npow, namelist="&dmclrdmc")
        else:
            self.lrdmc.comment_out(parameter="typereg")
            self.lrdmc.comment_out(parameter="npow")

        if io_fort10.pp_flag:
            self.lrdmc.set_parameter(parameter="parcutg", value=1, namelist="&dmclrdmc")
        else:
            self.lrdmc.set_parameter(parameter="parcutg", value=2, namelist="&dmclrdmc")

        if not self.force_calc_flag:
            pass
        else:
            self.lrdmc.set_parameter(
                parameter="ieskin", value=1, namelist="&parameters"
            )
            if self.io_fort10.pbc_flag:
                self.lrdmc.set_parameter(
                    parameter="typedyncell", value=2, namelist="&parameters"
                )
                self.lrdmc.set_parameter(
                    parameter="yespress", value='.true.', namelist="&parameters"
                )

        if pw_regularization > 0.0:
            #self.lrdmc.set_parameter(parameter="parcutg", value=0, namelist="&dmclrdmc")
            self.lrdmc.set_parameter(
                parameter="true_wagner", value=1, namelist="&dmclrdmc"
            )
            self.lrdmc.set_parameter(
                parameter="cutweight", value=-pw_regularization, namelist="&dmclrdmc"
            )

        # pseudo integration
        self.lrdmc.set_parameter(parameter="npsamax", value=4, namelist="&pseudo")

        # kpoints
        if self.twist_average:  # not 0 (= not False)!!
            if self.twist_average == 1:  # True case, Monkhorst-Pack algorithm
                assert len(self.kpoints) == 6
                nkx, nky, nkz, kx, ky, kz = self.kpoints
                self.lrdmc.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                # self.lrdmc.set_parameter(parameter="yeswrite10", value=".true.", namelist="&optimization")
                self.lrdmc.set_parameter(
                    parameter="kp_type", value=1, namelist="&kpoints"
                )
                self.lrdmc.set_parameter(
                    parameter="nk1", value=nkx, namelist="&kpoints"
                )
                self.lrdmc.set_parameter(
                    parameter="nk2", value=nky, namelist="&kpoints"
                )
                self.lrdmc.set_parameter(
                    parameter="nk3", value=nkz, namelist="&kpoints"
                )
                self.lrdmc.set_parameter(parameter="k1", value=kx, namelist="&kpoints")
                self.lrdmc.set_parameter(parameter="k2", value=ky, namelist="&kpoints")
                self.lrdmc.set_parameter(parameter="k3", value=kz, namelist="&kpoints")
                self.lrdmc.set_parameter(
                    parameter="skip_equivalence",
                    value=".true.",
                    namelist="&kpoints",
                )
                self.lrdmc.set_parameter(
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
                self.lrdmc.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                # self.lrdmc.set_parameter(parameter="yeswrite10", value=".true.", namelist="&optimization")
                self.lrdmc.set_parameter(
                    parameter="kp_type", value=2, namelist="&kpoints"
                )
                self.lrdmc.set_parameter(
                    parameter="nk1", value=len(kpoints_up), namelist="&kpoints"
                )
                self.lrdmc.set_parameter(
                    parameter="double_kpgrid",
                    value=".true.",
                    namelist="&kpoints",
                )
                self.lrdmc.manual_kpoints = self.kpoints

            else:
                logger.error(
                    f"twist_average = {self.twist_average} is not implemented."
                )
                raise NotImplementedError

    def run_all(
        self,
        bin_block: int = 10,
        warmupblocks: int = 2,
        correcting_factor: int = 2,
        cont: bool = False,
        input_name: str = "datasfn.input",
        output_name: str = "out_fn",
    ) -> None:
        """
        Generate input files and run the command.

        Parameters
        ----------
        bin_block : int, optional
            Binning length, by default 10.
        warmupblocks : int, optional
            Number of disregarded blocks, by default 2.
        correcting_factor : int, optional
            Correcting factors, by default 2.
        cont : bool, optional
            If True, continuation run (i.e., iopt=0). If False, starting from
            scratch (i.e., iopt=1), by default False.
        input_name : str, optional
            Input file name, by default "datasfn.input".
        output_name : str, optional
            Output file name, by default "out_fn".
        """
        self.generate_input(cont=cont, input_name=input_name)
        self.run(input_name=input_name, output_name=output_name)
        self.compute_energy_and_forces(
            bin_block=bin_block,
            warmupblocks=warmupblocks,
            correcting_factor=correcting_factor,
        )

    def generate_input(
        self, cont: bool = False, input_name: str = "datasfn.input"
    ) -> None:
        """
        Generate input file.

        Parameters
        ----------
        cont : bool, optional
            If True, continuation run (i.e., iopt=0). If False, starting from
            scratch (i.e., iopt=1), by default False.
        input_name : str, optional
            Input file name, by default "datasfn.input".
        """
        if cont:
            self.lrdmc.set_parameter("iopt", 0, "&simulation")
        self.lrdmc.generate_input(input_name=input_name)

    def run(
        self, input_name: str = "datasfn.input", output_name: str = "out_fn"
    ) -> None:
        """
        Run the command.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "datasfn.input".
        output_name : str, optional
            Output file name, by default "out_fn".

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully.
        """
        self.lrdmc.run(input_name=input_name, output_name=output_name)
        flags = self.lrdmc.check_results(output_names=[output_name])
        assert all(flags)

    def store_result(
        self,
        bin_block: int = 10,
        warmupblocks: int = 2,
        correcting_factor: int = 2,
        output_names: Optional[list] = None,
        rerun: bool = False,
    ) -> None:
        """
        Store results.

        This procedure stores estimated_time_for_1_generation, energy, and
        energy_error. This method is needed for storing data and accessing
        them later.

        Parameters
        ----------
        bin_block : int, optional
            Binning length, by default 10.
        warmupblocks : int, optional
            Number of disregarded blocks, by default 2.
        correcting_factor : int, optional
            Correcting factors, by default 2.
        output_names : list, optional
            A list of output file names. If None, defaults to ["out_fn"],
            by default None.
        rerun : bool, optional
            If True, compute energy and force again even if there are energy
            and force files, by default False.
        """
        if output_names is None:
            output_names = ["out_fn"]
        self.estimated_time_for_1_generation = self.get_estimated_time_for_1_generation(
            output_names=output_names
        )
        self.energy, self.energy_error = self.lrdmc.get_energy(
            init=warmupblocks,
            correct=correcting_factor,
            bin=bin_block,
            rerun=rerun,
        )

    def compute_energy_and_forces(
        self,
        bin_block: int = 10,
        warmupblocks: int = 2,
        correcting_factor: int = 2,
        rerun: bool = False,
    ) -> None:
        """
        Compute energy and forces.

        Parameters
        ----------
        bin_block : int, optional
            Binning length, by default 10.
        warmupblocks : int, optional
            Number of disregarded blocks, by default 2.
        correcting_factor : int, optional
            Correcting factors, by default 2.
        rerun : bool, optional
            If True, compute energy and force again even if there are energy
            and force files, by default False.
        """
        self.energy, self.energy_error = self.lrdmc.get_energy(
            init=warmupblocks,
            correct=correcting_factor,
            bin=bin_block,
            rerun=rerun,
        )

    def get_estimated_time_for_1_generation(
        self, output_names: Optional[list] = None
    ) -> float:
        """
        Get estimated time for one generation.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names. If None, defaults to ["out_fn"],
            by default None.

        Returns
        -------
        float
            Estimated time for one generation.
        """
        if output_names is None:
            output_names = ["out_fn"]
        return self.lrdmc.get_estimated_time_for_1_generation(output_names=output_names)

    def check_results(self, output_names: Optional[list] = None) -> bool:
        """
        Check the result.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names to check. If None, defaults to
            ["out_fn"], by default None.

        Returns
        -------
        bool
            True if all the runs were successful, False if an error is
            detected in the files.
        """
        if output_names is None:
            output_names = ["out_fn"]
        return self.lrdmc.check_results(output_names=output_names)


if __name__ == "__main__":
    logger = getLogger("Turbo-Genius")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    os.chdir(os.path.join(turbo_genius_root, "tests", "lrdmc"))

    # moved to examples
