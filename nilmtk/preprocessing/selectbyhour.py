from __future__ import print_function, division
from warnings import warn
from ..node import Node
from ..utils import index_of_column_name

class SelectByHour(Node):

    """Select the measurements by hour
    """
    requirements = {'device': {'measurements': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'selectbyhour': {}}}
    
    def __init__(self, hour=0):
        self.hour = hour

    def reset(self):
        self.hour = None

    def process(self):
        self.check_requirements()
        #metadata = self.upstream.get_metadata()
        #measurements = metadata['device']['measurements']
        for chunk in self.upstream.process():
            hour_mask = (chunk.index.hour==self.hour)
            new_chunk = chunk[hour_mask]
            new_chunk.timeframe = chunk.timeframe
            yield new_chunk