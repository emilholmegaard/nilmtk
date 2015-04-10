from __future__ import print_function, division
from warnings import warn
from ..node import Node
from ..utils import index_of_column_name
from _ctypes import ArgumentError

class SelectByHourAndDay(Node):

    """Select the measurements by hour
    """
    requirements = {'device': {'measurements': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'selectbyhourandday': {}}}
    
    def __init__(self, hour=0, day='B'):
        self.hour = hour
        self.day = day

    def reset(self):
        self.hour = None
        self.day = None

    def process(self):
        self.check_requirements()
        #metadata = self.upstream.get_metadata()
        #measurements = metadata['device']['measurements']
       
        for chunk in self.upstream.process():
            if self.day == 'B':
                #Saturday=0, Sunday=0
                businessday_mask = ((chunk.index.weekday< 6) | (chunk.index.weekday>0)) & (chunk.index.hour==self.hour)
                new_chunk = chunk[businessday_mask] 
            else:
                weekend_mask = ((chunk.index.weekday==6) | (chunk.index.weekday==0)) & (chunk.index.hour==self.hour)
                new_chunk = chunk[weekend_mask] 
                
            new_chunk.timeframe = chunk.timeframe
            yield new_chunk