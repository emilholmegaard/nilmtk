from __future__ import print_function, division
from ..node import Node


class SelectByExactDay(Node):

    """Select the measurements by hour
    """
    requirements = {'device': {'measurements': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'selectbyhourandday': {}}}
    
    def __init__(self, day=1):
        if day < 0 | day > 6:
            day = 1
        self.day = day

    def reset(self):
        self.day = None

    def process(self):
        self.check_requirements()
        #metadata = self.upstream.get_metadata()
        #measurements = metadata['device']['measurements']
       
        for chunk in self.upstream.process():
            mask = ((chunk.index.weekday == self.day))
            new_chunk = chunk[mask]                 
                
            new_chunk.timeframe = chunk.timeframe
            yield new_chunk