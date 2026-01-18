#!python -u
# -*- coding: utf-8 -*-

"""

GeniusIO abstract class

ToDo:
    To mix manual (by hand) and automatic (turbogenius) calculations,
    a "parse" function should be implemented in the geniusIO (and its children).
    Indeed, a function to purse an input file to generate the corresponding genius
    instance and the pickled file is needed to edit (by turbogenius) input and output files
    generated manually. This is also important for turboworkflow pakcage.

"""

# python modules
from abc import ABC, abstractmethod

# set logger
from logging import getLogger

logger = getLogger("Turbo-Genius").getChild(__name__)


# GeniusIO abstract class
class GeniusIO(ABC):
    """
    Abstract base class for Turbo-Genius I/O operations.

    This class defines the interface that all Turbo-Genius classes must implement.
    It provides a standardized way to generate input files, run calculations,
    and check results.

    Notes
    -----
    To mix manual (by hand) and automatic (turbogenius) calculations,
    a "parse" function should be implemented in the geniusIO (and its children).
    Indeed, a function to parse an input file to generate the corresponding genius
    instance and the pickled file is needed to edit (by turbogenius) input and output files
    generated manually. This is also important for turboworkflow package.

    Methods
    -------
    run_all()
        Generate input files and run the command.
    generate_input()
        Generate input file.
    run()
        Run the command.
    check_results()
        Check the result.
    """

    def __init__(
        self,
    ):
        """
        Initialize the GeniusIO abstract class.

        This is an abstract base class, so this method does nothing.
        Subclasses should override this method to initialize their specific attributes.
        """
        pass

    # abstract methods
    @abstractmethod
    def run_all(self):
        """
        Generate input files and run the command.

        This method should generate the necessary input files and then execute
        the calculation. It is a convenience method that combines generate_input()
        and run().

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate_input(self):
        """
        Generate input file.

        This method should create the input file(s) required for the calculation.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run the command.

        This method should execute the calculation using the generated input files.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def check_results(self):
        """
        Check the result.

        This method should verify that the calculation completed successfully
        and check for any errors in the output files.

        Returns
        -------
        bool
            True if all the runs were successful, False if an error is detected
            in the files.

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        """
        pass
