#!python -u
# -*- coding: utf-8 -*-

"""

pyturbo: fortranIO abstract class

"""

# python modules
from abc import ABC, abstractmethod

# set logger
from logging import getLogger

logger = getLogger("pyturbo").getChild(__name__)


# fortranIO abstract class
class FortranIO(ABC):
    """
    Abstract base class for TurboRVB Fortran program wrappers.

    This class provides a common interface for all TurboRVB program wrappers,
    including methods for parameter management, input generation, execution,
    and result checking.

    Attributes
    ----------
    namelist : Namelist
        Fortran namelist object containing program parameters.

    Notes
    -----
    Subclasses must implement all abstract methods to provide specific
    functionality for each TurboRVB program.
    """

    def __init__(
        self,
        # namelist=Namelist()
    ):
        """
        Initialize the FortranIO abstract class.

        This is an abstract base class, so this method does nothing.
        Subclasses should override this method to initialize their specific attributes.
        """
        # self.namelist = namelist
        pass

    # commom methods
    def set_parameter(self, parameter, value, namelist=None):
        """
        Set a parameter value in the namelist.

        Parameters
        ----------
        parameter : str
            Parameter name to set.
        value : int, float, or str
            Parameter value.
        namelist : str, optional
            Namelist name. If None, searches all namelists, by default None.

        Raises
        ------
        KeyError
            If the parameter is not found in the namelist.
        """
        self.namelist.set_parameter(parameter=parameter, value=value, namelist=namelist)

    def get_parameter(self, parameter, namelist=None):
        """
        Get a parameter value from the namelist.

        Parameters
        ----------
        parameter : str
            Parameter name to get.
        namelist : str, optional
            Namelist name. If None, searches all namelists, by default None.

        Returns
        -------
        int, float, or str
            Parameter value.

        Raises
        ------
        ValueError
            If the parameter is not found in the namelist.
        """
        return self.namelist.get_parameter(parameter=parameter, namelist=namelist)

    def comment_out(self, parameter):
        """
        Comment out a parameter in the namelist.

        Parameters
        ----------
        parameter : str
            Parameter name to comment out.

        Notes
        -----
        This method removes the parameter from the active namelist and adds
        it with a "!" prefix to comment it out.
        """
        self.namelist.comment_out(parameter=parameter)

    def get_parameters(self):
        """
        Get all parameters from the namelist.

        Returns
        -------
        dict
            Dictionary containing all namelists and their parameters.
        """
        return self.namelist.get_parameters()

    # abstract methods
    @abstractmethod
    def sanity_check(self):
        """
        Perform sanity checks on the input parameters.

        This method should validate that all required parameters are set
        correctly and that the configuration is valid for execution.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate_input(self):
        """
        Generate input file for the TurboRVB program.

        This method should create the input file(s) required for the
        TurboRVB program execution.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run the TurboRVB program.

        This method should execute the TurboRVB program with the generated
        input files.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def check_results(self):
        """
        Check the results of the TurboRVB program execution.

        This method should verify that the program completed successfully
        and check for any errors in the output files.

        Returns
        -------
        list of bool
            List of boolean flags indicating success for each output file.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def read_default_namelist(self):
        """
        Read default namelist values from the turbogenius database.

        Returns
        -------
        Namelist
            Namelist object with default parameter values.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def read_namelist_from_file(file):
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

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @classmethod
    @abstractmethod
    def parse_from_default_namelist(cls):
        """
        Create an instance with default namelist values.

        Returns
        -------
        cls
            Instance of the class with default namelist values.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @classmethod
    @abstractmethod
    def parse_from_file(cls):
        """
        Create an instance from a namelist file.

        Parameters
        ----------
        file : str
            Path to the input file.

        Returns
        -------
        cls
            Instance of the class with namelist values from the file.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass
