#!python
# -*- coding: utf-8 -*-
"""

lrdmc_opt genius related classes and methods

Todo:
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.

"""

# python modules
import os
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbogenius modules
from turbogenius.pyturbo.lrdmcopt import LRDMCopt
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.utils_workflows.env import turbo_genius_root
from turbogenius.utils_workflows.utility import (
    get_optimizer_flags,
    get_nonlocalmoves_setting,
)
from turbogenius.geniusIO import GeniusIO

logger = getLogger("Turbo-Genius").getChild(__name__)


class LRDMCopt_genius(GeniusIO):
    """
    Wrapper class for pyturbo LRDMCopt (LRDMC optimization) functionality.

    This class provides a high-level interface to optimize wavefunction
    parameters using LRDMC, with support for various optimization strategies
    and parameter types.

    Parameters
    ----------
    fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10".
    lrdmcoptsteps : int, optional
        Total number of optimization steps, by default 100.
    steps : int, optional
        Number of MCMC steps per optimization step, by default 10.
    bin_block : int, optional
        Binning length, by default 1.
    warmupblocks : int, optional
        Number of disregarded blocks, by default 1.
    num_walkers : int, optional
        Number of walkers. If -1, uses the number of MPI processes,
        by default -1.
    maxtime : int, optional
        Maximum time in seconds, by default 172800.
    optimizer : str, optional
        Optimizer choice: "sr" (stochastic reconfiguration) or "lr" (linear method),
        by default "sr".
    learning_rate : float, optional
        Optimization step size. Default values: sr:0.05, lr:0.35, by default 0.02.
    regularization : float, optional
        Regularization parameter, by default 0.001.
    num_opt_param : int, optional
        Number of optimized parameters. 0 means all parameters are optimized,
        by default 0.
    alat : float, optional
        Lattice space in Bohr, by default -0.20.
    time_branching : float, optional
        Interval between two branching steps in atomic units, by default 0.10.
    etry : float, optional
        Trial energy in Hartree, by default 0.0.
    nonlocalmoves : str, optional
        Treatment of locality approximation. Choose from "tmove", "dla", "dlatm",
        by default "dla".
    opt_onebody : bool, optional
        Flag to optimize one-body Jastrow, by default True.
    opt_twobody : bool, optional
        Flag to optimize two-body Jastrow, by default True.
    opt_det_mat : bool, optional
        Flag to optimize matrix elements in the determinant part, by default False.
    opt_jas_mat : bool, optional
        Flag to optimize matrix elements in the Jastrow part, by default True.
    opt_det_basis_exp : bool, optional
        Flag to optimize exponents of the determinant basis sets, by default False.
    opt_jas_basis_exp : bool, optional
        Flag to optimize exponents of the Jastrow basis sets, by default False.
    opt_det_basis_coeff : bool, optional
        Flag to optimize coefficients of the determinant basis sets, by default False.
    opt_jas_basis_coeff : bool, optional
        Flag to optimize coefficients of the Jastrow basis sets, by default False.
    twist_average : bool, optional
        Twist average flag, True or False, by default False.
    kpoints : list, optional
        k Monkhorst-Pack grids, [kx,ky,kz,nx,ny,nz], where kx,y,z are grids
        and nx,y,z are shift (0) or no shift (1), by default [1, 1, 1, 0, 0, 0].

    Attributes
    ----------
    fort10 : str
        Input fort.10 wavefunction file.
    lrdmcopt : LRDMCopt
        Underlying pyturbo LRDMCopt instance.
    energy : float or None
        Energy value (set after calculation).
    energy_error : float or None
        Energy error (set after calculation).
    estimated_time_for_1_generation : float or None
        Estimated time for one generation (set after calculation).
    """

    def __init__(
        self,
        fort10: str = "fort.10",
        lrdmcoptsteps: int = 100,
        steps: int = 10,
        bin_block: int = 1,
        warmupblocks: int = 1,
        num_walkers: int = -1,  # default -1 -> num of MPI process.
        maxtime: int = 172800,
        optimizer: str = "sr",
        learning_rate: float = 0.02,
        regularization: float = 0.001,
        num_opt_param: int = 0,
        alat: float = -0.20,
        time_branching: float = 0.10,
        etry: float = 0.0,
        nonlocalmoves: str = "dla",  # tmove, dla, dlatm
        opt_onebody: bool = True,
        opt_twobody: bool = True,
        opt_det_mat: bool = False,
        opt_jas_mat: bool = True,
        opt_det_basis_exp: bool = False,
        opt_jas_basis_exp: bool = False,
        opt_det_basis_coeff: bool = False,
        opt_jas_basis_coeff: bool = False,
        twist_average: bool = False,
        kpoints: Optional[list] = None,
    ):

        if kpoints is None:
            kpoints = [1, 1, 1, 0, 0, 0]

        self.fort10 = fort10
        self.twist_average = twist_average
        self.kpoints = kpoints

        self.energy = None
        self.energy_error = None
        self.estimated_time_for_1_generation = None

        (
            optimizer_number,
            iesdonebodyoff,
            iesdtwobodyoff,
            twobodyoff,
            iesd,
            iesfree,
            iessw,
            iesup,
            iesm,
        ) = get_optimizer_flags(
            optimizer=optimizer,
            opt_onebody=opt_onebody,
            opt_twobody=opt_twobody,
            opt_det_mat=opt_det_mat,
            opt_jas_mat=opt_jas_mat,
            opt_det_basis_exp=opt_det_basis_exp,
            opt_jas_basis_exp=opt_jas_basis_exp,
            opt_det_basis_coeff=opt_det_basis_coeff,
            opt_jas_basis_coeff=opt_jas_basis_coeff,
            qmc_type="lrdmc",
        )

        self.lrdmcopt = LRDMCopt.parse_from_default_namelist(
            in_fort10=fort10, twist_average=twist_average
        )

        self.lrdmcopt.set_parameter(
            parameter="itestr4", value=optimizer_number, namelist="&simulation"
        )
        self.lrdmcopt.set_parameter(
            parameter="ngen",
            value=lrdmcoptsteps * steps,
            namelist="&simulation",
        )
        self.lrdmcopt.set_parameter(
            parameter="maxtime", value=maxtime, namelist="&simulation"
        )
        if num_walkers != -1:
            self.lrdmcopt.set_parameter(
                parameter="nw", value=num_walkers, namelist="&simulation"
            )
        self.lrdmcopt.set_parameter(
            parameter="nweight", value=steps, namelist="&optimization"
        )
        self.lrdmcopt.set_parameter(
            parameter="nbinr", value=bin_block, namelist="&optimization"
        )
        self.lrdmcopt.set_parameter(
            parameter="iboot", value=warmupblocks, namelist="&optimization"
        )
        self.lrdmcopt.set_parameter(
            parameter="tpar", value=learning_rate, namelist="&optimization"
        )
        self.lrdmcopt.set_parameter(
            parameter="parr", value=regularization, namelist="&optimization"
        )
        if num_opt_param !=0:
            self.lrdmcopt.set_parameter(
                parameter="npbra", value=num_opt_param, namelist="&optimization"
            )
        self.lrdmcopt.set_parameter(
            parameter="iesdonebodyoff",
            value=iesdonebodyoff,
            namelist="&optimization",
        )
        self.lrdmcopt.set_parameter(
            parameter="iesdtwobodyoff",
            value=iesdtwobodyoff,
            namelist="&optimization",
        )
        self.lrdmcopt.set_parameter(
            parameter="twobodyoff", value=twobodyoff, namelist="&optimization"
        )

        self.lrdmcopt.set_parameter(
            parameter="iesd", value=iesd, namelist="&parameters"
        )
        self.lrdmcopt.set_parameter(
            parameter="iesfree", value=iesfree, namelist="&parameters"
        )
        self.lrdmcopt.set_parameter(
            parameter="iessw", value=iessw, namelist="&parameters"
        )
        self.lrdmcopt.set_parameter(
            parameter="iesup", value=iesup, namelist="&parameters"
        )
        self.lrdmcopt.set_parameter(
            parameter="iesm", value=iesm, namelist="&parameters"
        )

        self.lrdmcopt.set_parameter(parameter="alat", value=alat, namelist="&dmclrdmc")
        self.lrdmcopt.set_parameter(parameter="etry", value=etry, namelist="&dmclrdmc")
        self.lrdmcopt.set_parameter(
            parameter="tbra", value=time_branching, namelist="&dmclrdmc"
        )

        typereg, npow = get_nonlocalmoves_setting(nonlocalmoves=nonlocalmoves)
        self.lrdmcopt.set_parameter(
            parameter="typereg", value=typereg, namelist="&dmclrdmc"
        )
        self.lrdmcopt.set_parameter(parameter="npow", value=npow, namelist="&dmclrdmc")

        # regularization
        # to be arguments of the class
        self.lrdmcopt.set_parameter(parameter="parcutg", value=0, namelist="&dmclrdmc")
        self.lrdmcopt.set_parameter(
            parameter="true_wagner", value=1, namelist="&dmclrdmc"
        )
        self.lrdmcopt.set_parameter(
            parameter="cutweight", value=-1.0e-4, namelist="&dmclrdmc"
        )

        # pseudo integration
        self.lrdmcopt.set_parameter(parameter="npsamax", value=4, namelist="&pseudo")

        # kpoints
        if self.twist_average:  # not 0 (= not False)!!
            if self.twist_average == 1:  # True case, Monkhorst-Pack algorithm
                assert len(self.kpoints) == 6
                nkx, nky, nkz, kx, ky, kz = self.kpoints
                self.lrdmcopt.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                # self.lrdmcopt.set_parameter(parameter="yeswrite10", value=".true.", namelist="&optimization")
                self.lrdmcopt.set_parameter(
                    parameter="kp_type", value=1, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="nk1", value=nkx, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="nk2", value=nky, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="nk3", value=nkz, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="k1", value=kx, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="k2", value=ky, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="k3", value=kz, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="skip_equivalence",
                    value=".true.",
                    namelist="&kpoints",
                )
                self.lrdmcopt.set_parameter(
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
                self.lrdmcopt.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                # self.lrdmcopt.set_parameter(parameter="yeswrite10", value=".true.", namelist="&optimization")
                self.lrdmcopt.set_parameter(
                    parameter="kp_type", value=2, namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="nk1", value=len(kpoints_up), namelist="&kpoints"
                )
                self.lrdmcopt.set_parameter(
                    parameter="double_kpgrid",
                    value=".true.",
                    namelist="&kpoints",
                )
                self.lrdmcopt.manual_kpoints = self.kpoints

            else:
                logger.error(
                    f"twist_average = {self.twist_average} is not implemented."
                )
                raise NotImplementedError

    def run_all(
        self,
        cont: bool = False,
        input_name: str = "datasfn_opt.input",
        output_name: str = "out_fn_opt",
        average_parameters: bool = True,
    ) -> None:
        """
        Generate input files and run the command.

        Parameters
        ----------
        cont : bool, optional
            If True, continuation run (i.e., iopt=0). If False, starting from
            scratch (i.e., iopt=1), by default False.
        input_name : str, optional
            Input file name, by default "datasfn_opt.input".
        output_name : str, optional
            Output file name, by default "out_fn_opt".
        average_parameters : bool, optional
            If True, average the optimized parameters, by default True.
        """
        self.generate_input(cont=cont, input_name=input_name)
        self.run(input_name=input_name, output_name=output_name)
        if average_parameters:
            self.average(input_name=input_name, output_name=output_name)

    def generate_input(
        self, cont: bool = False, input_name: str = "datasfn_opt.input"
    ) -> None:
        """
        Generate input file.

        Parameters
        ----------
        cont : bool, optional
            If True, continuation run (i.e., iopt=0). If False, starting from
            scratch (i.e., iopt=1), by default False.
        input_name : str, optional
            Input file name, by default "datasfn_opt.input".
        """
        io_fort10 = IO_fort10(fort10=self.fort10)
        io_fort10.io_flag = 0
        if cont:
            self.lrdmcopt.set_parameter("iopt", 0, "&simulation")
        self.lrdmcopt.generate_input(input_name=input_name)

    def run(
        self,
        input_name: str = "datasfn_opt.input",
        output_name: str = "out_fn_opt",
    ) -> None:
        """
        Run the command.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "datasfn_opt.input".
        output_name : str, optional
            Output file name, by default "out_fn_opt".

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully.
        """
        self.lrdmcopt.run(input_name=input_name, output_name=output_name)
        flags = self.lrdmcopt.check_results(output_names=[output_name])
        assert all(flags)

    def store_result(self, output_names: Optional[list] = None) -> None:
        """
        Store results.

        Energy, energy_error, and estimated_time_for_1_generation are stored
        in this class.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names. If None, defaults to ["out_fn_opt"],
            by default None.
        """
        if output_names is None:
            output_names = ["out_fn_opt"]
        self.energy, self.energy_error = self.get_energy(output_names=output_names)
        self.estimated_time_for_1_generation = self.get_estimated_time_for_1_generation(
            output_names=output_names
        )

    def plot_energy_and_devmax(
        self,
        output_names: Optional[list] = None,
        interactive: bool = True,
    ):
        """
        Plot energy and devmax.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names. If None, defaults to ["out_fn_opt"],
            by default None.
        interactive : bool, optional
            Flag for an interactive plot, by default True.
        """
        if output_names is None:
            output_names = ["out_fn_opt"]
        self.lrdmcopt.plot_energy_and_devmax(
            output_names=output_names, interactive=interactive
        )

    def average(
        self,
        optwarmupsteps: int = 10,
        graph_plot: bool = False,
        input_name: str = "datasfn_opt.input",
        output_names: Optional[list] = None,
    ) -> None:
        """
        Average parameters of fort.10.

        Parameters
        ----------
        optwarmupsteps : int, optional
            Number of disregarded optimization steps, by default 10.
        graph_plot : bool, optional
            Flag for plotting a graph, by default False.
        input_name : str, optional
            Input file used in the latest calculation, by default "datasfn_opt.input".
        output_names : list, optional
            A list of output file names. If None, defaults to ["out_fn_opt"],
            by default None.

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully.
        """
        if output_names is None:
            output_names = ["out_fn_opt"]
        flags = self.lrdmcopt.check_results(output_names=output_names)
        assert all(flags)
        self.lrdmcopt.average_optimized_parameters(
            equil_steps=optwarmupsteps,
            input_file_used=input_name,
            graph_plot=graph_plot,
        )

    def get_energy(self, output_names: Optional[list] = None) -> list:
        """
        Get energy list.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names. If None, defaults to ["out_fn_opt"],
            by default None.

        Returns
        -------
        list
            A list of history of energies.
        """
        if output_names is None:
            output_names = ["out_fn_opt"]
        return self.lrdmcopt.get_energy(output_names=output_names)

    def get_estimated_time_for_1_generation(
        self, output_names: Optional[list] = None
    ) -> float:
        """
        Get estimated time for one generation.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names. If None, defaults to ["out_fn_opt"],
            by default None.

        Returns
        -------
        float
            Estimated time for one generation.
        """
        if output_names is None:
            output_names = ["out_fn_opt"]
        return self.lrdmcopt.get_estimated_time_for_1_generation(
            output_names=output_names
        )

    def check_results(self, output_names: Optional[list] = None) -> bool:
        """
        Check the result.

        Parameters
        ----------
        output_names : list, optional
            A list of output file names to check. If None, defaults to
            ["out_fn_opt"], by default None.

        Returns
        -------
        bool
            True if all the runs were successful, False if an error is
            detected in the files.
        """
        if output_names is None:
            output_names = ["out_fn_opt"]
        return self.lrdmcopt.check_results(output_names=output_names)

    def plot_parameters_history(self, interactive: bool = True) -> None:
        """
        Plot history of optimized variational parameters.

        Parameters
        ----------
        interactive : bool, optional
            Flag for an interactive plot, by default True.
        """
        self.lrdmcopt.plot_parameters_history(interactive=interactive)


if __name__ == "__main__":
    logger = getLogger("Turbo-Genius")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    os.chdir(os.path.join(turbo_genius_root, "tests", "lrdmcopt"))

    lrdmcopt_genius = LRDMCopt_genius(
        fort10="fort.10",
        lrdmcoptsteps=100,
        optwarmupsteps=10,
        steps=500,
        bin_block=2,
        warmupblocks=1,
        optimizier="sr",
        learning_rate=0.02,
        regularization=0.001,
        alat=-0.20,
        etry=-1.0,
        nonlocalmoves="tmove",
        opt_onebody=True,
        opt_twobody=True,
        opt_det_mat=False,
        opt_jas_mat=False,
        opt_det_basis_exp=False,
        opt_jas_basis_exp=False,
        opt_det_basis_coeff=False,
        opt_jas_basis_coeff=False,
        twist_average=False,
    )
    lrdmcopt_genius.run()
