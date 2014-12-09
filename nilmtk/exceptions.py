"""File defining custom nilmtk exception classes.
"""

class TooFewSamplesError(Exception):
    pass

class UnableToGetWeatherData(Exception):
    pass

class WrongResolution(Exception):
    pass