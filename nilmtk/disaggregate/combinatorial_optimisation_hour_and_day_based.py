from __future__ import print_function, division
import pandas as pd
import numpy as np
import json
from datetime import datetime
from warnings import warn
from ..appliance import ApplianceID
from ..utils import find_nearest, find_nearest_2,container_to_string
from ..feature_detectors import cluster
from ..timeframe import merge_timeframes, list_of_timeframe_dicts, TimeFrame
from ..preprocessing import Apply, Clip, SelectByHour, SelectByHourAndDay

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)
printing = True

class COHourDay(object):

    """1 dimensional combinatorial optimisation NILM algorithm.

    Attributes
    ----------
    model : dict for each hour, with a list of dicts of the model
       Each dict has these keys:
           states : list of ints (the power (Watts) used in different states)
           training_metadata : ElecMeter or MeterGroup object used for training 
               this set of states.  We need this information because we 
               need the appliance type (and perhaps some other metadata)
               for each model.
    """

    def __init__(self):
        self.model = dict()
        self.timespan = range(0,24,1)
        self.day_types = ['B', 'WEEKEND']
        for day_type in self.day_types:
            for t in self.timespan:
                self.model[t,day_type]=[]
        
        #self.model = []

    def train(self, metergroup):
        """Train using 1D CO. Places the learnt model in the `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object

        Notes
        -----
        * only uses first chunk for each meter (TODO: handle all chunks).
        """

        for day_type in self.day_types:
            for t in self.timespan:
                if self.model[t,day_type]:
                    raise RuntimeError("This implementation of Combinatorial Optimisation"
                               " does not support multiple calls to `train`.")

        num_meters = len(metergroup.meters)
        if num_meters > 12:
            max_num_clusters = 2
        else:
            max_num_clusters = 3

        for i, meter in enumerate(metergroup.submeters().meters):
            if printing:
                print("Training model for submeter '{}'".format(meter))
            #Business day or Weekend
            for day_type in self.day_types:
                #For each hour
                for t in self.timespan:
                    try:
                        for chunk in meter.power_series(preprocessing=[SelectByHourAndDay(hour=t, day=day_type)]):
                        #for chunk in meter.power_series(preprocessing=[Clip(), SelectByHourAndDay(hour=t, day=day_type)]):
                            states = cluster(chunk, max_num_clusters)
                            self.model[t,day_type].append({
                                   'states': states,
                                   'training_metadata': meter})
                            break  # TODO handle multiple chunks per appliance
                    except:
                        warn('Problem training the model in hour {}.'.format(t, day_type), RuntimeWarning)
                 
        if printing:
            print("Done training!")
            print(self.model)

    def disaggregate(self, mains, output_datastore, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        output_name : string, optional
            The `name` to use in the metadata for the `output_datastore`.
            e.g. some sort of name for this experiment.  Defaults to 
            "NILMTK_CO_<date>"
        resample_seconds : number, optional
            The desired sample period in seconds.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        MIN_CHUNK_LENGTH = 100

        for day_type in self.day_types:
            for t in self.timespan:
                if not self.model[t,day_type]:
                    raise RuntimeError("The model needs to be instantiated before"
                               " calling `disaggregate`.  For example, the"
                               " model can be instantiated by running `train`.")

        # If we import sklearn at the top of the file then auto doc fails.
        from sklearn.utils.extmath import cartesian

        # sklearn produces lots of DepreciationWarnings with PyTables
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Extract optional parameters from load_kwargs
        date_now = datetime.now().isoformat().split('.')[0]
        output_name = load_kwargs.pop('output_name', 'NILMTK_CO_' + date_now)
        resample_seconds = load_kwargs.pop('resample_seconds', 60)
        
        centroids = dict()
        state_combinations = dict()
        vampire_power_array = dict()
        summed_power_of_each_combination = dict()
        indices_of_state_combinations=dict()
        residual_power=dict()
        timeframes=dict()
                    
        for day_type in self.day_types:
            for t in self.timespan:
    
                # Get centroids
                centroids[t,day_type] = [m['states'] for m in self.model[t,day_type]]
                state_combinations[t,day_type] = cartesian(centroids[t,day_type])
                # state_combinations is a 2D array
                # each column is a chan
                # each row is a possible combination of power demand values e.g.
                # [[0, 0, 0, 0], [0, 0, 0, 100], [0, 0, 50, 0], [0, 0, 50, 100], ...]
        
                # Add vampire power to the model
                '''
                vampire_power = 0
                vampire_power = mains.vampire_power()
                if printing:
                    print("vampire_power = {} watts".format(vampire_power))
                n_rows = state_combinations[t].shape[0]
                vampire_power_array[t] = np.zeros((n_rows, 1)) + vampire_power
                state_combinations[t] = np.hstack((state_combinations[t], vampire_power_array[t]))
                '''
                vampire_power = 0
                n_rows = state_combinations[t,day_type].shape[0]
                vampire_power_array[t,day_type] = np.zeros((n_rows, 1)) + vampire_power
                state_combinations[t,day_type] = np.hstack((state_combinations[t,day_type], vampire_power_array[t,day_type]))
                summed_power_of_each_combination[t,day_type] = np.sum(state_combinations[t,day_type], axis=1)
                # summed_power_of_each_combination is now an array where each
                # value is the total power demand for each combination of states.
        
                #load_kwargs['sections'] = load_kwargs.pop('sections', mains.good_sections())
                
                load_kwargs['preprocessing']=[SelectByHourAndDay(hour=t, day=day_type)]
                
               
                
                resample_rule = '{:d}S'.format(resample_seconds)
                timeframes[t,day_type] = []
                building_path = '/building{}'.format(mains.building())
                mains_data_location = '{}/elec/meter1'.format(building_path)
                if True:
                #try:
                    for chunk in mains.power_series(**load_kwargs):
            
                        # Check that chunk is sensible size before resampling
                        if len(chunk) < MIN_CHUNK_LENGTH:
                            continue
            
                        # Record metadata
                        timeframes[t,day_type].append(chunk.timeframe)
                        measurement = chunk.name
            
                        chunk = chunk.resample(rule=resample_rule)#, how='mean', fill_method='bfill')
                        # Check chunk size *again* after resampling
                        if len(chunk) < MIN_CHUNK_LENGTH:
                            continue
            
                        # Start disaggregation
                        indices_of_state_combinations[t,day_type], residual_power[t,day_type] = find_nearest(
                            summed_power_of_each_combination[t,day_type], chunk.values)
                        if printing:
                            print("Hour: {} {}".format(t,day_type))
                            print("MIN and MAX of summed_power_of_each_combination: {} and {}".format(np.min(summed_power_of_each_combination[t,day_type]),np.max(summed_power_of_each_combination[t,day_type])))
                            print("MIN and MAX of chunk.values: {} and {}".format(np.nanmin(chunk.values),np.nanmax(chunk.values)))
                        
                        
                        for i, m in enumerate(self.model[t,day_type]):
                            if printing:
                                print("Estimating power demand for '{}'".format(m['training_metadata']))
                            predicted_power = state_combinations[t,day_type][indices_of_state_combinations[t,day_type], i].flatten()
                            cols = pd.MultiIndex.from_tuples([chunk.name])
                            meter_instance = m['training_metadata'].instance()
                            output_datastore.append('{}/elec/meter{}'
                                                        .format(building_path, meter_instance),
                                                        pd.DataFrame(predicted_power,
                                                                     index=chunk.index,
                                                                     columns=cols))
                else:
                #except:
                    warn('Problem training the model in hour {}.'.format(t,day_type), RuntimeWarning)
    
                cols = pd.MultiIndex.from_tuples([chunk.name])
                # Copy mains data to disag output
                output_datastore.append(key=mains_data_location,
                                        value=pd.DataFrame(chunk, columns=cols))

        ##################################
        # Add metadata to output_datastore

        # TODO: `preprocessing_applied` for all meters
        # TODO: split this metadata code into a separate function
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        
        #Add metadata for main meter
        mains_meter = mains.metadata['device_model'] if hasattr(mains, 'metadata') else 'mains'
        
        meter_devices = {
            'CO': {
                'model': 'CO',
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'mains': {
                'model': mains_meter,
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            }
        }

        merged_timeframes = merge_timeframes(timeframes[0, 'B'], gap=resample_seconds)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        dataset_metadata = {'name': output_name, 'date': date_now,
                            'meter_devices': meter_devices,
                            'timeframe': total_timeframe.to_dict()}
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                'device_model': mains_meter,
                'site_meter': True,
                'data_location': mains_data_location,
                'preprocessing_applied': {},  # TODO
                'statistics': {
                    'timeframe': total_timeframe.to_dict(),
                    'good_sections': list_of_timeframe_dicts(merged_timeframes)
                }
            }
        }

        # Appliances and submeters:
        appliances = []
        #Potential error here by using hour 0, 'B'
        for model in self.model[0, 'B']:
            meter = model['training_metadata']

            meter_instance = meter.instance()

            for app in meter.appliances:
                meters = app.metadata['meters']
                appliance = {
                    'meters': [meter_instance], 
                    'type': app.identifier.type,
                    'instance': app.identifier.instance
                    # TODO this `instance` will only be correct when the
                    # model is trained on the same house as it is tested on.
                    # https://github.com/nilmtk/nilmtk/issues/194
                }
                appliances.append(appliance)

            elec_meters.update({
                meter_instance: {
                    'device_model': 'CO',
                    'submeter_of': 1,
                    'data_location': ('{}/elec/meter{}'
                                      .format(building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict(),
                        'good_sections': list_of_timeframe_dicts(merged_timeframes)
                    }
                }
            })

        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

    # TODO: fix export and import!
    # https://github.com/nilmtk/nilmtk/issues/193
    #
    # def export_model(self, filename):
    #     model_copy = {}
    #     for appliance, appliance_states in self.model.iteritems():
    #         model_copy[
    #             "{}_{}".format(appliance.name, appliance.instance)] = appliance_states
    #     j = json.dumps(model_copy)
    #     with open(filename, 'w+') as f:
    #         f.write(j)

    # def import_model(self, filename):
    #     with open(filename, 'r') as f:
    #         temp = json.loads(f.read())
    #     for appliance, centroids in temp.iteritems():
    #         appliance_name = appliance.split("_")[0].encode("ascii")
    #         appliance_instance = int(appliance.split("_")[1])
    #         appliance_name_instance = ApplianceID(
    #             appliance_name, appliance_instance)
    #         self.model[appliance_name_instance] = centroids
