from __future__ import print_function, division
from ..node import Node


class SelectByDay(Node):

    """Select the measurements by hour
    """
    requirements = {'device': {'measurements': 'ANY VALUE'}}
    postconditions =  {'preprocessing_applied': {'selectbyhourandday': {}}}
    
    def __init__(self, day='B'):
        self.day = day

    def reset(self):
        self.day = None

    def process(self):
        self.check_requirements()
        #metadata = self.upstream.get_metadata()
        #measurements = metadata['device']['measurements']
       
        for chunk in self.upstream.process():
            if self.day == 'B':
                #Saturday=0, Sunday=0
                businessday_mask = ((chunk.index.weekday <= 5) & (chunk.index.weekday >= 1))
                new_chunk = chunk[businessday_mask]                 
            else:
                weekend_mask = ((chunk.index.weekday > 5)  | (chunk.index.weekday < 1))
                new_chunk = chunk[weekend_mask] 
                
            new_chunk.timeframe = chunk.timeframe
            yield new_chunk