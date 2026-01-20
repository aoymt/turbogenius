#!python
# -*- coding: utf-8 -*-

"""

vmc_opt genius related classes and methods

Todo:
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.

"""

# python modules
import os
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbogenius modules
from turbogenius.pyturbo.vmcopt import VMCopt
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.utils_workflows.env import turbo_genius_root
from turbogenius.utils_workflows.utility import get_optimizer_flags
from turbogenius.tools_genius import copy_jastrow_twist
from turbogenius.geniusIO import GeniusIO

logger = getLogger("Turbo-Genius").getChild(__name__)


class VMCopt_genius(GeniusIO):
    """
    Wrapper class for pyturbo VMCopt (VMC optimization) functionality.

    This class provides a high-level interface to optimize wavefunction
    parameters using VMC, with support for various optimization strategies
    and parameter types.

    Parameters
    ----------
    fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10".
    vmcoptsteps : int, optional
        Total number of optimization steps, by default 100.
    steps : int, optional
        Number of MCMC steps per optimization step, by default 10.
    bin_block : int, optional
        Binning length, by default 1.
    warmupblocks : int, optional
        Number of disregarded blocks, by default 0.
    num_walkers : int, optional
        Number of walkers. If -1, uses the number of MPI processes,
        by default -1.
    maxtime : int, optional
        Maximum time in seconds, by default 172800.
    optimizer : str, optional
        Optimizer choice: "sr" (stochastic reconfiguration) or "lr" (linear method),
        by default "sr".
    learning_rate : float, optional
        Optimization step size. Default values: sr:0.05, lr:0.35, by default 0.35.
    regularization : float, optional
        Regularization parameter, by default 0.001.
    num_opt_param : int, optional
        Number of optimized parameters. 0 means all parameters are optimized,
        by default 0.
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
    opt_structure : bool, optional
        Flag to optimize the structure, by default False.
    str_learning_rate : float, optional
        Optimization step size for structural optimization, by default 1.0e-6.
    twist_average : bool, optional
        Twist average flag, True or False, by default False.
    kpoints : list, optional
        k Monkhorst-Pack grids, [kx,ky,kz,nx,ny,nz], where kx,y,z are grids
        and nx,y,z are shift (0) or no shift (1), by default [1, 1, 1, 0, 0, 0].

    Attributes
    ----------
    fort10 : str
        Input fort.10 wavefunction file.
    vmcopt : VMCopt
        Underlying pyturbo VMCopt instance.
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
        vmcoptsteps: int = 100,
        steps: int = 10,
        bin_block: int = 1,
        warmupblocks: int = 0,
        num_walkers: int = -1,  # default -1 -> num of MPI process.
        maxtime: int = 172800,
        optimizer: str = "sr",
        learning_rate: float = 0.35,
        regularization: float = 0.001,
        num_opt_param: int = 0,
        opt_onebody: bool = True,
        opt_twobody: bool = True,
        opt_det_mat: bool = False,
        opt_jas_mat: bool = True,
        opt_det_basis_exp: bool = False,
        opt_jas_basis_exp: bool = False,
        opt_det_basis_coeff: bool = False,
        opt_jas_basis_coeff: bool = False,
        opt_structure: bool = False,
        str_learning_rate: float = 1.0e-6,
        twist_average: bool = False,
        kpoints: Optional[list] = None,
    ):

        if kpoints is None:
            kpoints = [1, 1, 1, 0, 0, 0]

        self.fort10 = fort10
        self.twist_average = twist_average
        self.kpoints = kpoints

        self.optimizer = optimizer
        self.opt_onebody = opt_onebody
        self.opt_twobody = opt_twobody
        self.opt_det_mat = opt_det_mat
        self.opt_jas_mat = opt_jas_mat
        self.opt_det_basis_exp = opt_det_basis_exp
        self.opt_jas_basis_exp = opt_jas_basis_exp
        self.opt_det_basis_coeff = opt_det_basis_coeff
        self.opt_jas_basis_coeff = opt_jas_basis_coeff

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
            qmc_type="vmc",
        )

        self.energy = None
        self.energy_error = None
        self.estimated_time_for_1_generation = None

        self.vmcopt = VMCopt.parse_from_default_namelist(
            in_fort10=fort10, twist_average=self.twist_average
        )

        self.vmcopt.set_parameter(
            parameter="itestr4", value=optimizer_number, namelist="&simulation"
        )
        self.vmcopt.set_parameter(
            parameter="ngen", value=vmcoptsteps * steps, namelist="&simulation"
        )
        self.vmcopt.set_parameter(
            parameter="maxtime", value=maxtime, namelist="&simulation"
        )
        if num_walkers != -1:
            self.vmcopt.set_parameter(
                parameter="nw", value=num_walkers, namelist="&simulation"
            )

        # pseudo integration
        self.vmcopt.set_parameter(
            parameter="npsamax", value=4, namelist="&pseudo"
        )

        # vmc optimization
        self.vmcopt.set_parameter(
            parameter="nweight", value=steps, namelist="&optimization"
        )
        self.vmcopt.set_parameter(
            parameter="nbinr", value=bin_block, namelist="&optimization"
        )
        self.vmcopt.set_parameter(
            parameter="iboot", value=warmupblocks, namelist="&optimization"
        )
        self.vmcopt.set_parameter(
            parameter="tpar", value=learning_rate, namelist="&optimization"
        )
        self.vmcopt.set_parameter(
            parameter="parr", value=regularization, namelist="&optimization"
        )

        self.vmcopt.set_parameter(
            parameter="iesdonebodyoff",
            value=iesdonebodyoff,
            namelist="&optimization",
        )
        self.vmcopt.set_parameter(
            parameter="iesdtwobodyoff",
            value=iesdtwobodyoff,
            namelist="&optimization",
        )
        self.vmcopt.set_parameter(
            parameter="twobodyoff", value=twobodyoff, namelist="&optimization"
        )

        self.vmcopt.set_parameter(
            parameter="iesd", value=iesd, namelist="&parameters"
        )
        self.vmcopt.set_parameter(
            parameter="iesfree", value=iesfree, namelist="&parameters"
        )
        self.vmcopt.set_parameter(
            parameter="iessw", value=iessw, namelist="&parameters"
        )
        self.vmcopt.set_parameter(
            parameter="iesup", value=iesup, namelist="&parameters"
        )
        self.vmcopt.set_parameter(
            parameter="iesm", value=iesm, namelist="&parameters"
        )

        if num_opt_param !=0:
            self.vmcopt.set_parameter(
                parameter="npbra", value=num_opt_param, namelist="&optimization"
            )

        # structural optimization
        if opt_structure:
            logger.info("Structural optimization flag is on.")
            self.vmcopt.set_parameter(
                parameter="ieskin", value=1, namelist="&parameters"
            )
            self.vmcopt.set_parameter(
                parameter="idyn", value=5, namelist="&optimization"
            )
            self.vmcopt.set_parameter(
                parameter="tion",
                value=str_learning_rate,
                namelist="&optimization",
            )
            self.vmcopt.set_parameter(
                parameter="temp", value=0.0, namelist="&dynamic"
            )
            self.vmcopt.set_parameter(
                parameter="iskipdyn", value=5, namelist="&dynamic"
            )
            self.vmcopt.set_parameter(
                parameter="maxdev_dyn", value=6.0, namelist="&dynamic"
            )
            self.vmcopt.set_parameter(
                parameter="ngen",
                value=vmcoptsteps * steps * 5,
                namelist="&simulation",
            )  # 5 = iskipdyn

        # Does the VMC optimization changes the nodal surface? if not, it is better to switch off epscut option.
        if opt_det_mat or opt_det_basis_exp or opt_det_basis_coeff:
            self.vmcopt.comment_out(parameter="epscut")
        else:
            self.vmcopt.set_parameter(
                parameter="epscut", value=0.0, namelist="&vmc"
            )

        # kpoints
        if self.twist_average:  # not 0 (= not False)!!
            if self.twist_average == 1:  # True case, Monkhorst-Pack algorithm
                assert len(self.kpoints) == 6
                nkx, nky, nkz, kx, ky, kz = self.kpoints
                self.vmcopt.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                # self.vmcopt.set_parameter(parameter="yeswrite10", value=".true.", namelist="&optimization")
                self.vmcopt.set_parameter(
                    parameter="kp_type", value=1, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="nk1", value=nkx, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="nk2", value=nky, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="nk3", value=nkz, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="k1", value=kx, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="k2", value=ky, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="k3", value=kz, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="skip_equivalence",
                    value=".true.",
                    namelist="&kpoints",
                )
                self.vmcopt.set_parameter(
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
                self.vmcopt.set_parameter(
                    parameter="yes_kpoints",
                    value=".true.",
                    namelist="&parameters",
                )
                # self.vmcopt.set_parameter(parameter="yeswrite10", value=".true.", namelist="&optimization")
                self.vmcopt.set_parameter(
                    parameter="kp_type", value=2, namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="nk1", value=len(kpoints_up), namelist="&kpoints"
                )
                self.vmcopt.set_parameter(
                    parameter="double_kpgrid",
                    value=".true.",
                    namelist="&kpoints",
                )
                self.vmcopt.manual_kpoints = self.kpoints

            else:
                logger.error(
                    f"twist_average = {self.twist_average} is not implemented."
                )
                raise NotImplementedError

    def run_all(
        self,
        optwarmsteps: int,
        cont: bool = False,
        input_name: str = "datasmin.input",
        output_name: str = "out_min",
        average_parameters: bool = True,
    ) -> None:
        """
        Generate input files and run the VMC optimization.

        Parameters
        ----------
        optwarmsteps : int
            Number of disregarded optimization steps.
        cont : bool, optional
            If True, continuation run (i.e., iopt=0). If False, starting from
            scratch (i.e., iopt=1), by default False.
        input_name : str, optional
            Input file name, by default "datasmin.input".
        output_name : str, optional
            Output file name, by default "out_min".
        average_parameters : bool, optional
            If True, average the optimized parameters after optimization,
            by default True.

        Notes
        -----
        This method calls generate_input(), run(), and optionally average()
        to complete the VMC optimization process.
        """
        self.generate_input(cont=cont, input_name=input_name)
        self.run(input_name=input_name, output_name=output_name)
        if average_parameters:
            self.average(
                optwarmupsteps=optwarmsteps,
                input_name=input_name,
                output_names=[output_name],
            )

    def generate_input(
        self, cont: bool = False, input_name: str = "datasmin.input"
    ) -> None:
        """
        Generate input file for the VMCopt program.

        Parameters
        ----------
        cont : bool, optional
            If True, continuation run (i.e., iopt=0). If False, starting from
            scratch (i.e., iopt=1), by default False.
        input_name : str, optional
            Input file name, by default "datasmin.input".

        Notes
        -----
        This method generates the datasmin.input file based on the parameters
        set during initialization.
        """
        io_fort10 = IO_fort10(fort10=self.fort10)
        io_fort10.io_flag = 0
        if cont:
            self.vmcopt.set_parameter("iopt", 0, "&simulation")
        self.vmcopt.generate_input(input_name=input_name)

    def run(
        self, input_name: str = "datasmin.input", output_name: str = "out_min"
    ) -> None:
        """
        Run the VMCopt program.

        Parameters
        ----------
        input_name : str, optional
            Input file name, by default "datasmin.input".
        output_name : str, optional
            Output file name, by default "out_min".

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully (check_results
            indicates failure for any output file).

        Notes
        -----
        This method executes the VMCopt program and then checks the results.
        An AssertionError is raised if any output file indicates failure.
        """
        self.vmcopt.run(input_name=input_name, output_name=output_name)
        flags = self.vmcopt.check_results(output_names=[output_name])
        assert all(flags)

    def check_results(self, output_names: Optional[list] = None) -> bool:
        """
        Check the results of the VMCopt program execution.

        Parameters
        ----------
        output_names : list, optional
            List of output file names to check. If None, defaults to
            ["out_min"], by default None.

        Returns
        -------
        bool
            True if all the runs were successful, False if an error is
            detected in the output files.

        Notes
        -----
        This method checks the output files for successful completion
        by looking for specific patterns indicating successful execution.
        """
        if output_names is None:
            output_names = ["out_min"]
        return self.vmcopt.check_results(output_names=output_names)

    def plot_energy_and_devmax(
        self, output_names: Optional[list] = None, interactive: bool = True
    ) -> None:
        """
        Plot energy and devmax as a function of optimization steps.

        Parameters
        ----------
        output_names : list, optional
            List of output file names. If None, defaults to ["out_min"],
            by default None.
        interactive : bool, optional
            If True, display an interactive plot. If False, save to file,
            by default True.

        Notes
        -----
        This method creates plots showing the convergence of energy and
        devmax during the optimization process.
        """
        if output_names is None:
            output_names = ["out_min"]
        self.vmcopt.plot_energy_and_devmax(
            output_names=output_names, interactive=interactive
        )

    def store_result(self, output_names: Optional[list] = None) -> None:
        """
        Store calculation results in instance attributes.

        Energy, energy_error, and estimated_time_for_1_generation are stored
        in this class.

        Parameters
        ----------
        output_names : list, optional
            List of output file names. If None, defaults to ["out_min"],
            by default None.

        Notes
        -----
        The stored values can be accessed via self.energy, self.energy_error,
        and self.estimated_time_for_1_generation attributes.
        """
        if output_names is None:
            output_names = ["out_min"]
        self.energy, self.energy_error = self.get_energy(
            output_names=output_names
        )
        self.estimated_time_for_1_generation = (
            self.get_estimated_time_for_1_generation(output_names=output_names)
        )

    def average(
        self,
        optwarmupsteps: int,
        input_name: str = "datasmin.input",
        output_names: Optional[list] = None,
        graph_plot: bool = False,
    ) -> None:
        """
        Average optimized parameters and update fort.10.

        Parameters
        ----------
        optwarmupsteps : int
            Number of disregarded optimization steps.
        input_name : str, optional
            Input file used in the latest calculation, by default "datasmin.input".
        output_names : list, optional
            List of output file names. If None, defaults to ["out_min"],
            by default None.
        graph_plot : bool, optional
            If True, plot a graph of parameter convergence, by default False.

        Raises
        ------
        AssertionError
            If the calculation does not complete successfully.
        NotImplementedError
            If twist averaging with certain optimization flags is not supported.

        Notes
        -----
        This method averages the optimized parameters over the last steps
        (excluding warmup steps) and updates the fort.10 file. For twist-averaged
        calculations with JDFT ansatz, it also copies Jastrow factors.
        """
        if output_names is None:
            output_names = ["out_min"]
        current_dir = os.getcwd()
        os.chdir(current_dir)
        if self.twist_average:
            if (
                self.opt_det_mat
                or self.opt_det_basis_exp
                or self.opt_det_basis_coeff
            ):
                logger.warning(
                    "The twist average with a JAGP ansatz, turbogenius assumes real_agp=.true. option."
                )
                twist_average_copyjas = False
            elif self.opt_jas_basis_exp or self.opt_jas_basis_coeff:
                logger.warning(
                    "Sorry opt_jas_basis_exp or opt_jas_basis_coeff are not supported."
                )
                raise NotImplementedError
            else:  # self.opt_jas_mat:
                logger.info(
                    "The twist average with a JDFT ansatz (i.e., optimize only jas. mat.) is now supported."
                )
                twist_average_copyjas = True
        else:
            logger.info(
                "Twist-average flag is False. Open boundary condition or PBC at single k point."
            )
            twist_average_copyjas = False

        flags = self.vmcopt.check_results(output_names=output_names)
        assert all(flags)
        self.vmcopt.average_optimized_parameters(
            equil_steps=optwarmupsteps,
            input_file_used=input_name,
            graph_plot=graph_plot,
        )

        if twist_average_copyjas:
            copy_jastrow_twist()

            """
            logger.info("Additional commands are needed for averaging Jas. mat. with k average")
            logger.info("cp fort.10 turborvb.scratch/fort.10;")
            logger.info("cp fort.10 turborvb.scratch/fort.10_new;")
            logger.info("cp kp_info.dat turborvb.scratch/kp_info.dat;")
            logger.info("cp parminimized.d turborvb.scratch/parminimized.d;")
            logger.info("cd turborvb.scratch/")
            logger.info("copyjas.x kpoints")

            shutil.copy("fort.10", os.path.join("turborvb.scratch", "fort.10"))
            shutil.copy("fort.10", os.path.join("turborvb.scratch", "fort.10_new"))
            shutil.copy("kp_info.dat", os.path.join("turborvb.scratch", "kp_info.dat"))
            shutil.copy("parminimized.d", os.path.join("turborvb.scratch", "parminimized.d"))
            os.chdir("turborvb.scratch")
            copy_jastrow(twist_flag=True)
            os.chdir(current_dir)
            """

    def get_energy(self, output_names: Optional[list] = None) -> list:
        """
        Get energy history from optimization output files.

        Parameters
        ----------
        output_names : list, optional
            List of output file names. If None, defaults to ["out_min"],
            by default None.

        Returns
        -------
        list
            List containing [energy, energy_error] extracted from the output files.

        Notes
        -----
        This method reads the energy values from the optimization output files.
        """
        if output_names is None:
            output_names = ["out_min"]
        return self.vmcopt.get_energy(output_names=output_names)

    def get_estimated_time_for_1_generation(
        self, output_names: Optional[list] = None
    ) -> float:
        """
        Get estimated time for one generation from output files.

        Parameters
        ----------
        output_names : list, optional
            List of output file names. If None, defaults to ["out_min"],
            by default None.

        Returns
        -------
        float
            Estimated time for one generation in seconds.

        Notes
        -----
        This method reads the output files and extracts the average time
        for 1000 generations, then divides by 1000 to get the time per generation.
        """
        if output_names is None:
            output_names = ["out_min"]
        return self.vmcopt.get_estimated_time_for_1_generation(
            output_names=output_names
        )

    def plot_parameters_history(self, interactive: bool = True) -> None:
        """
        Plot history of optimized variational parameters.

        Parameters
        ----------
        interactive : bool, optional
            If True, display an interactive plot. If False, save to file,
            by default True.

        Notes
        -----
        This method creates plots showing the evolution of variational parameters
        during the optimization process.
        """
        self.vmcopt.plot_parameters_history(interactive=interactive)


if __name__ == "__main__":
    logger = getLogger("Turbo-Genius")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter(
        "%(name)s - %(levelname)s - %(lineno)d - %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    os.chdir(os.path.join(turbo_genius_root, "tests", "vmcopt"))
    # removed to examples
