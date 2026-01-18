#!python
# -*- coding: utf-8 -*-

"""

pyturbo: prep related classes and methods

Todo:
    * docstrings are not completed.
    * refactoring assert sentences. The assert should not be used for any on-the-fly check.
    * implementing __str__ method.
    * implementing sanity_check method.

"""

# python modules
import os
import re
import numpy as np
from typing import Optional

# pyturbo modules
from turbogenius.pyturbo.namelist import Namelist
from turbogenius.pyturbo.fortranIO import FortranIO
from turbogenius.pyturbo.utils.env import pyturbo_data_dir
from turbogenius.pyturbo.utils.env import turbo_prep_run_command
from turbogenius.pyturbo.utils.utility import file_check
from turbogenius.pyturbo.utils.execute import run
from turbogenius.pyturbo.io_fort10 import IO_fort10

from logging import getLogger, StreamHandler, Formatter

logger = getLogger("pyturbo").getChild(__name__)


class Prep(FortranIO):
    """
    Wrapper class for TurboRVB prep program.

    This class provides an interface to the prep program, which is used for
    DFT calculations and preparing wavefunctions for QMC calculations.

    Parameters
    ----------
    in_fort10 : str, optional
        Input fort.10 wavefunction file, by default "fort.10".
    namelist : Namelist, optional
        Namelist object containing program parameters. If None, an empty
        Namelist is created, by default None.
    nelocc_list : list, optional
        List of occupation numbers for up electrons, by default [].
    neloccdn_list : list, optional
        List of occupation numbers for down electrons, by default [].
    magnetic_moments_3d_array : list or numpy.ndarray, optional
        3D array of magnetic moments with shape (nzs, nys, nxs), by default [].
    twist_average : bool or int, optional
        Twist average flag. False or 0: single-k point, True or 1: Monkhorst-Pack,
        2: manual k-grid, by default False.

    Attributes
    ----------
    in_fort10 : str
        Input fort.10 wavefunction file.
    namelist : Namelist
        Namelist object containing program parameters.
    nelocc_list : list
        List of occupation numbers for up electrons.
    neloccdn_list : list
        List of occupation numbers for down electrons.
    magnetic_moments_3d_array : numpy.ndarray
        3D array of magnetic moments.
    twist_average : bool or int
        Twist average flag.
    manual_kpoints : list
        Manual k-points list (set via property setter).

    Examples
    --------
    See tests directory for usage examples.
    """

    def __init__(
        self,
        in_fort10: str = "fort.10",
        namelist: Optional[Namelist] = None,
        nelocc_list: Optional[list] = None,
        neloccdn_list: Optional[list] = None,
        magnetic_moments_3d_array: Optional[
            list
        ] = None,  # dim = 3, shape = (nzs,nys,nxs)
        twist_average: bool = False,  # False or 0: single-k, True or 1: Monkhorst-Pack, 2: manual k-grid
    ):
        """
        Initialize the Prep class.

        Parameters
        ----------
        in_fort10 : str, optional
            Input fort.10 wavefunction file, by default "fort.10".
        namelist : Namelist, optional
            Namelist object containing program parameters. If None, an empty
            Namelist is created, by default None.
        nelocc_list : list, optional
            List of occupation numbers for up electrons, by default [].
        neloccdn_list : list, optional
            List of occupation numbers for down electrons, by default [].
        magnetic_moments_3d_array : list or numpy.ndarray, optional
            3D array of magnetic moments with shape (nzs, nys, nxs), by default [].
        twist_average : bool or int, optional
            Twist average flag. False or 0: single-k point, True or 1: Monkhorst-Pack,
            2: manual k-grid, by default False.

        Raises
        ------
        FileNotFoundError
            If the fort.10 file or pseudo.dat (if needed) is not found.
        """
        if namelist is None:
            namelist = Namelist()
        if nelocc_list is None:
            nelocc_list = []
        if neloccdn_list is None:
            neloccdn_list = []
        if magnetic_moments_3d_array is None:
            magnetic_moments_3d_array = []

        """
        input values
        """
        file_check(in_fort10)
        if IO_fort10(fort10=in_fort10).pp_flag:
            file_check("pseudo.dat")

        self.in_fort10 = in_fort10
        self.namelist = namelist
        self.nelocc_list = nelocc_list
        self.neloccdn_list = neloccdn_list
        self.magnetic_moments_3d_array = magnetic_moments_3d_array
        self.twist_average = twist_average

        # manual k-grid! [[[kx, ky, kz, wkp for up], ....], [[# kx, ky, kz, wkp for dn], ...]]
        self.__manual_kpoints = []

    @property
    def manual_kpoints(self):
        """
        Get manual k-points.

        Returns
        -------
        list
            List of manual k-points. Format: [[kpoints_up], [kpoints_dn]],
            where each kpoint is [kx, ky, kz, wkp].
        """
        return self.__manual_kpoints

    @manual_kpoints.setter
    def manual_kpoints(self, kpoints):
        """
        Set manual k-points.

        Parameters
        ----------
        kpoints : list
            List of manual k-points. Format: [[kpoints_up], [kpoints_dn]],
            where each kpoint is [kx, ky, kz, wkp].

        Raises
        ------
        AssertionError
            If the k-points format is invalid.

        Notes
        -----
        This setter automatically configures the namelist for manual k-points
        when twist_average == 2.
        """
        assert len(kpoints) == 2
        kpoints_up, kpoints_dn = kpoints
        assert len(kpoints_up) == len(kpoints_dn)
        for kup, kdn in zip(kpoints_up, kpoints_dn):
            assert len(kup) == 4  # kx, ky, kz, wkp for up
            assert len(kdn) == 4  # kx, ky, kz, wkp for dn
        self.__manual_kpoints = kpoints
        self.namelist.set_parameter(
            parameter="yes_kpoints", value=".true.", namelist="&parameters"
        )
        self.namelist.set_parameter(
            parameter="yeswrite10", value=".true.", namelist="&optimization"
        )
        self.namelist.set_parameter(parameter="kp_type", value=2, namelist="&kpoints")
        self.namelist.set_parameter(
            parameter="nk1", value=len(kpoints_up), namelist="&kpoints"
        )

    @property
    def magnetic_moments_3d_array(self):
        """
        Get magnetic moments 3D array.

        Returns
        -------
        numpy.ndarray
            3D array of magnetic moments with shape (nzs, nys, nxs).
        """
        return self.__magnetic_moments_3d_array

    @magnetic_moments_3d_array.setter
    def magnetic_moments_3d_array(self, matrix):
        """
        Set magnetic moments 3D array.

        Parameters
        ----------
        matrix : numpy.ndarray
            3D array of magnetic moments with shape (nzs, nys, nxs).

        Raises
        ------
        KeyError
            If nxs, nys, or nzs are not set in the namelist.
        AssertionError
            If the matrix dimensions or shape are invalid.
        """
        if len(matrix) != 0:
            try:
                nxs = self.namelist.get_parameter(parameter="nxs")
                nys = self.namelist.get_parameter(parameter="nys")
                nzs = self.namelist.get_parameter(parameter="nzs")
            except KeyError:
                logger.error("nxs, nys, or nzs are not correctly set in namelist!")
                raise KeyError
            assert matrix.ndim == 3
            assert matrix.shape == (nzs, nys, nxs)
        self.__magnetic_moments_3d_array = matrix

    def __str__(self):
        """
        Return string representation of the Prep object.

        Returns
        -------
        str
            String description of the object.
        """
        output = [
            "TurboRVB prep python wrapper",
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

    def generate_input(self, input_name: str = "prep.input"):
        """
        Generate input file for the prep program.

        Parameters
        ----------
        input_name : str, optional
            Output input file name, by default "prep.input".

        Raises
        ------
        ValueError
            If no suitable position for KPOINT is found when using manual k-points.

        Notes
        -----
        If twist_average == 2 (manual k-points), the k-points are inserted
        into the input file at the appropriate location.
        """
        self.namelist.write(input_name)
        # check if twist_average is manual
        if self.twist_average == 2:  # k-points are set manually
            kpoints_up, kpoints_dn = self.manual_kpoints
            # read
            with open(input_name, "r") as f:
                lines = f.readlines()
            kpoint_index = [
                True if re.match(r".*&kpoints.*", line) else False for line in lines
            ].index(True)
            insert_lineno = -1
            for i, line in enumerate(lines[kpoint_index + 1 :]):
                if re.match(r".*/.*", line):
                    insert_lineno = kpoint_index + 1 + i + 1
                    break
                if re.match(r".*&.*", line):
                    insert_lineno = kpoint_index + 1 + i
                    break
            if insert_lineno == -1:
                logger.error("No suitable position for KPOINT is found")
                raise ValueError
            # write dn
            lines.insert(insert_lineno, "\n")
            for kx, ky, kz, wk in reversed(kpoints_dn):
                lines.insert(insert_lineno, f"{kx} {ky} {kz} {wk}\n")
            lines.insert(insert_lineno, "\n")
            # write up
            for kx, ky, kz, wk in reversed(kpoints_up):
                lines.insert(insert_lineno, f"{kx} {ky} {kz} {wk}\n")
            lines.insert(insert_lineno, "KPOINTS\n")
            lines.insert(insert_lineno, "\n")
            # saved
            with open(input_name, "w") as f:
                f.writelines(lines)
        # write occupation list
        logger.debug(self.nelocc_list)
        logger.debug(self.neloccdn_list)
        if len(self.nelocc_list) != 0:
            with open(input_name, "a") as f:
                line = " ".join(list(map(str, self.nelocc_list))) + "\n"
                f.write(line)
        if len(self.neloccdn_list) != 0:
            with open(input_name, "a") as f:
                line = "\n" + " ".join(list(map(str, self.neloccdn_list))) + "\n"
                f.write(line)
        if len(self.magnetic_moments_3d_array) != 0:
            # write magnetic fields
            logger.info("Writing magnetic moments at the end of prep.input file ... \n")
            with open(input_name, "a") as f:
                f.write("\n")
                nzs = self.namelist.get_parameter(parameter="nzs")
                for z_index in range(nzs):
                    np.savetxt(f, self.magnetic_moments_3d_array[z_index, :, :], "%d")
                    f.write("\n")
            logger.info("Magnetic moments written. \n")
        logger.info(f"{input_name} has been generated. \n")

    def run(self, input_name="prep.input", output_name="out_prep"):
        run(
            turbo_prep_run_command,
            input_name=input_name,
            output_name=output_name,
        )

    def check_results(self, output_names: Optional[list] = None):
        if output_names is None:
            output_names = ["out_prep"]
        flags = []
        for output_name in output_names:
            file_check(output_name)
            with open(output_name, "r") as f:
                lines = f.readlines()
            if any(
                [
                    re.match(
                        r".*OK.*Turbo-DFT.*converged.*with.*energy.*tollerance.*",
                        line,
                    )
                    for line in lines
                ]
            ):
                flags.append(True)
            else:
                flags.append(False)
        return flags

    @staticmethod
    def read_default_namelist(in_fort10: str = "fort.10", det_contraction_flag=None):
        prep_default_file = os.path.join(pyturbo_data_dir, "prep", "prep.input")
        namelist = Namelist.parse_namelist_from_file(prep_default_file)

        # fort.10
        io_fort10 = IO_fort10(fort10=in_fort10)
        # ! right! we should keep io_fort10 "local".
        # It should not be an attrobute because it could be very large..
        
        if det_contraction_flag is None:
            det_contraction_flag = io_fort10.det_contraction_flag
        # contraction
        if det_contraction_flag:
            namelist.set_parameter(
                parameter="contracted_on", value=".true.", namelist="&dft"
            )
        else:
            namelist.set_parameter(
                parameter="contracted_on", value=".false.", namelist="&dft"
            )

        # occ
        typedft = namelist.get_parameter(parameter="typedft")
        optocc = namelist.get_parameter(parameter="optocc")
        # io_fort10 = IO_fort10(fort10=in_fort10) # not needed!!
        if optocc == 0:
            if typedft == 1:  # lda
                nelocc = io_fort10.f10header.nelup
                namelist.set_parameter(
                    parameter="nelocc", value=nelocc, namelist="&dft"
                )
            elif typedft == 4:  # 4
                nelocc = io_fort10.f10header.nelup
                neloccdn = io_fort10.f10header.neldn
                namelist.set_parameter(
                    parameter="nelocc", value=nelocc, namelist="&dft"
                )
                namelist.set_parameter(
                    parameter="neloccdo", value=neloccdn, namelist="&dft"
                )

        return namelist

    @staticmethod
    def read_namelist_from_file(file):
        namelist = Namelist.parse_namelist_from_file(file)
        return namelist

    @classmethod
    def parse_from_default_namelist(
        cls,
        in_fort10: str = "fort.10",
        magnetic_moments_3d_array: Optional[list] = None,
        twist_average: bool = False,
        det_contraction_flag: Optional[bool] = None
    ):
        if magnetic_moments_3d_array is None:
            magnetic_moments_3d_array = []
        namelist = cls.read_default_namelist(in_fort10=in_fort10, det_contraction_flag=det_contraction_flag)
        # fort10
        io_fort10 = IO_fort10(fort10=in_fort10)
        # occ
        typedft = namelist.get_parameter(parameter="typedft")
        optocc = namelist.get_parameter(parameter="optocc")
        if optocc == 0:
            if typedft == 1:  # lda
                nelocc_list = [2 for _ in range(io_fort10.f10header.neldn)] + [
                    1
                    for _ in range(
                        io_fort10.f10header.nelup - io_fort10.f10header.neldn
                    )
                ]
                neloccdn_list = []
            elif typedft == 4:  # 4
                nelocc_list = [1 for _ in range(io_fort10.f10header.nelup)]
                neloccdn_list = [1 for _ in range(io_fort10.f10header.neldn)]

        else:
            nelocc_list = []
            neloccdn_list = []

        return cls(
            in_fort10=in_fort10,
            namelist=namelist,
            nelocc_list=nelocc_list,
            neloccdn_list=neloccdn_list,
            magnetic_moments_3d_array=magnetic_moments_3d_array,
            twist_average=twist_average,
        )

    @classmethod
    def parse_from_file(cls, file: str, in_fort10: str = "fort.10"):
        namelist = Namelist.parse_namelist_from_file(file)
        logger.warning(f"nelocc_list and neloccdn_list are not read from {file}")
        logger.warning(f"magnetic_moments_3d_array is not read from {file}")
        logger.warning(f"KPOINTS is not read from {file}")
        return cls(in_fort10=in_fort10, namelist=namelist)


if __name__ == "__main__":
    logger = getLogger("pyturbo")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # moved to examples
