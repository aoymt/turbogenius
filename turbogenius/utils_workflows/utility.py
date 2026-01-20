#!python -u
# -*- coding: utf-8 -*-

from __future__ import print_function

# set logger
from logging import getLogger

logger = getLogger("pyturbo").getChild(__name__)


def prompt(text, checker):
    """
    Loop input() until valid input is provided.

    Parameters
    ----------
    text : str
        Prompt text to display to the user.
    checker : callable
        Function that takes the input string and returns True if valid,
        False otherwise.

    Returns
    -------
    str
        Valid user input string.

    Notes
    -----
    This function will loop indefinitely until the checker function
    returns True for the user input.
    """
    while True:
        output = input(text)
        if checker(output):
            return output


def get_nonlocalmoves_setting(nonlocalmoves: str):
    """
    Get nonlocal moves setting parameters for LRDMC calculations.

    Parameters
    ----------
    nonlocalmoves : str
        Type of nonlocal moves treatment. Options: "tmove", "dla", "dlatm", "la".

    Returns
    -------
    tuple
        A tuple containing (typereg, npow) where:
        - typereg (int): Type of regularization.
        - npow (float): Power parameter.

    Raises
    ------
    NotImplementedError
        If the specified nonlocalmoves type is not implemented.

    Notes
    -----
    Available options:

    - "tmove": T-move algorithm (typereg=0, npow=0.0)
    - "dla": DLA algorithm (typereg=6, npow=1.0)
    - "dlatm": DLA-TM algorithm (typereg=6, npow=0.0)
    - "la": Locality approximation (typereg=0, npow=1.0)
    """
    if nonlocalmoves == "tmove":
        typereg = 0
        npow = 0.0
    elif nonlocalmoves == "dla":
        typereg = 6
        npow = 1.0
    elif nonlocalmoves == "dlatm":
        typereg = 6
        npow = 0.0
    elif nonlocalmoves == "la":
        typereg = 0
        npow = 1.0
    else:
        logger.error(f"nonlocalmoves={nonlocalmoves} is not implemented.")
        raise NotImplementedError

    return typereg, npow


def get_optimizer_flags(
    optimizer: str = "sr",
    opt_onebody: bool = True,
    opt_twobody: bool = True,
    opt_det_mat: bool = True,
    opt_jas_mat: bool = True,
    opt_det_basis_exp: bool = True,
    opt_jas_basis_exp: bool = True,
    opt_det_basis_coeff: bool = True,
    opt_jas_basis_coeff: bool = True,
    qmc_type: str = "vmc",
):
    """
    Get optimizer flags for QMC calculations.

    This function determines the appropriate flags for various optimization
    options in VMC or LRDMC calculations.

    Parameters
    ----------
    optimizer : str, optional
        Optimizer type: "sr" (stochastic reconfiguration) or "lr" (linear method),
        by default "sr".
    opt_onebody : bool, optional
        Flag to optimize one-body Jastrow, by default True.
    opt_twobody : bool, optional
        Flag to optimize two-body Jastrow, by default True.
    opt_det_mat : bool, optional
        Flag to optimize determinant matrix elements, by default True.
    opt_jas_mat : bool, optional
        Flag to optimize Jastrow matrix elements, by default True.
    opt_det_basis_exp : bool, optional
        Flag to optimize determinant basis set exponents, by default True.
    opt_jas_basis_exp : bool, optional
        Flag to optimize Jastrow basis set exponents, by default True.
    opt_det_basis_coeff : bool, optional
        Flag to optimize determinant basis set coefficients, by default True.
    opt_jas_basis_coeff : bool, optional
        Flag to optimize Jastrow basis set coefficients, by default True.
    qmc_type : str, optional
        QMC calculation type: "vmc" or "lrdmc", by default "vmc".

    Returns
    -------
    tuple
        A tuple containing:
        - optimizer_number (int): Optimizer number for the calculation.
        - iesdonebodyoff (str): Flag for one-body Jastrow optimization (".true." or ".false.").
        - iesdtwobodyoff (str): Flag for two-body Jastrow optimization (".true." or ".false.").
        - twobodyoff (str): Flag for two-body Jastrow off (".true." or ".false.").
        - iesd (int): Flag for Jastrow optimization (0 or 1).
        - iesfree (int): Flag for Jastrow matrix optimization (0 or 1).
        - iessw (int): Flag for determinant matrix optimization (0 or 1).
        - iesup (int): Flag for determinant basis set optimization (0 or 1).
        - iesm (int): Flag for Jastrow basis set optimization (0 or 1).

    Raises
    ------
    NotImplementedError
        If the optimizer type is not supported.
    AssertionError
        If there are conflicts in optimization flags.

    Notes
    -----
    The function automatically determines the appropriate optimizer number
    based on the combination of optimization flags and QMC type.
    """
    logger.debug(optimizer)

    # check if either onebody or twobody jastrow is optimized
    if opt_onebody or opt_twobody:
        iesd = 1
        if opt_onebody:
            iesdonebodyoff = ".false."
        else:
            iesdonebodyoff = ".true."
        if opt_twobody:
            iesdtwobodyoff = ".false."
        else:
            iesdtwobodyoff = ".true."
    else:
        iesd = 0
        iesdonebodyoff = ".false."
        iesdtwobodyoff = ".false."

    # jastrow matrix element optimization
    if opt_jas_mat:
        iesfree = 1
        twobodyoff = ".false."
    else:
        if opt_onebody:
            iesfree = 1
            twobodyoff = ".true."
        else:
            iesfree = 0
            twobodyoff = ".false."

    # determinant matrix element optimization
    if opt_det_mat:
        iessw = 1
    else:
        iessw = 0

    # det. basis set optimization
    if opt_det_basis_coeff or opt_det_basis_exp:
        iesup = 1
        if opt_det_basis_coeff:
            if optimizer == "sr":  # stochastic reconfiguration
                optimizer_number = return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=True
                )
            elif optimizer == "lr":  # linear method
                optimizer_number = return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=True
                )
            else:
                raise NotImplementedError
        else:
            if optimizer == "sr":  # stochastic reconfiguration
                optimizer_number = return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
                )
            elif optimizer == "lr":  # linear method
                optimizer_number = return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
                )
            else:
                raise NotImplementedError
    else:
        iesup = 0
        if optimizer == "sr":  # stochastic reconfiguration
            optimizer_number = return_optimizer_number(
                optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
            )
        elif optimizer == "lr":  # linear method
            optimizer_number = return_optimizer_number(
                optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
            )
        else:
            raise NotImplementedError

    # jas. basis set optimization
    if opt_jas_basis_coeff or opt_jas_basis_exp:
        iesm = 1
        if opt_jas_basis_coeff:
            if optimizer == "sr":  # stochastic reconfiguration
                assert optimizer_number == return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=True
                ), "There is a conflict!! Not implemented."
            elif optimizer == "lr":  # linear method
                assert optimizer_number == return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=True
                ), "There is a conflict!! Not implemented."
            else:
                raise NotImplementedError
        else:
            if optimizer == "sr":  # stochastic reconfiguration
                assert optimizer_number == return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
                ), "There is a conflict!! Not implemented."
            elif optimizer == "lr":  # linear method
                assert optimizer_number == return_optimizer_number(
                    optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
                ), "There is a conflict!! Not implemented."
            else:
                raise NotImplementedError
    else:
        iesm = 0

        if optimizer == "sr":  # stochastic reconfiguration
            assert optimizer_number == return_optimizer_number(
                optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
            ), "There is a conflict!! Not implemented."
        elif optimizer == "lr":  # linear method
            assert optimizer_number == return_optimizer_number(
                optimizer=optimizer, qmc_type=qmc_type, opt_basis_coeff=False
            ), "There is a conflict!! Not implemented."
        else:
            raise NotImplementedError

    return (
        optimizer_number,
        iesdonebodyoff,
        iesdtwobodyoff,
        twobodyoff,
        iesd,
        iesfree,
        iessw,
        iesup,
        iesm,
    )


def return_optimizer_number(
    optimizer: str = "sr", qmc_type: str = "vmc", opt_basis_coeff: bool = False
):
    """
    Return optimizer number based on optimizer type, QMC type, and basis coefficient optimization.

    Parameters
    ----------
    optimizer : str, optional
        Optimizer type: "sr" (stochastic reconfiguration) or "lr" (linear method),
        by default "sr".
    qmc_type : str, optional
        QMC calculation type: "vmc" or "lrdmc", by default "vmc".
    opt_basis_coeff : bool, optional
        Flag indicating whether basis set coefficients are optimized, by default False.

    Returns
    -------
    int
        Optimizer number:
        - For "lr" optimizer:
          - VMC: -8 (with basis coeff) or -4 (without)
          - LRDMC: -28 (with basis coeff) or -24 (without)
        - For "sr" optimizer:
          - VMC: -5 (with basis coeff) or -9 (without)
          - LRDMC: -25 (with basis coeff) or -29 (without)

    Raises
    ------
    NotImplementedError
        If the optimizer or QMC type is not supported.
    """
    if optimizer == "lr":
        if qmc_type == "vmc":
            if opt_basis_coeff:
                optimizer_number = -8
            else:
                optimizer_number = -4
        elif qmc_type == "lrdmc":
            if opt_basis_coeff:
                optimizer_number = -28
            else:
                optimizer_number = -24
        else:
            raise NotImplementedError

    elif optimizer == "sr":
        if qmc_type == "vmc":
            if opt_basis_coeff:
                optimizer_number = -5
            else:
                optimizer_number = -9
        elif qmc_type == "lrdmc":
            if opt_basis_coeff:
                optimizer_number = -25
            else:
                optimizer_number = -29
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return optimizer_number
