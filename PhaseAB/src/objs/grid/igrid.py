import numpy as np
from abc import ABC, abstractmethod

class IGrid(ABC):
    """
    Interface for the Grid object. 

    Interfaces are used to define the methods that should be implemented by the classes that inherit from them.
    In simple terms, an interface is a blueprint for a class. It defines a set of methods that the class must implement.
    """
    @abstractmethod
    def create_grid(self):
        pass

    @abstractmethod
    def find_blocks(self, contours: list[np.ndarray]):
        pass

