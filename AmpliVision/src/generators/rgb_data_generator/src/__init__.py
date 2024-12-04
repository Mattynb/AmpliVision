from .utils import Utils
from .data_extractor import DataExtractor

__all__ = ['Utils', 'DataExtractor']


"""

The initial goal of this module was to generate random RGB data points based on extracted fingerprints.
We eventually realized the overfitting potential, so we decided to generate whole test images based on the fingerprints.

Most of the previous files were deleted, but some functions are still useful for other parts of AmpliVision.
"""