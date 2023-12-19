from fcrd_asset import FcrDDownAsset, FcrDAssetNoEnergyReservoir
from sequences import (SequenceOfRampsBuilder, SeqOfPoint2SeqOfRamps,
                       SequenceOfFrequencySignalsFollowingSineCurve, MergeSequences)

import time
import datetime as dt
import math

# These enums represent states in the execution of the tests.
# These are central to a state machine running.
# Central is the "without running state", which is the only state with
# transition to/from other states.
from enum import Enum
class RunState(Enum):
    have_no_tests_to_run = 0
    without_running_state = 1
    running_subtest = 2
    running_charge_or_discharge = 3

class BaseRunHolder:
    def __init__(self, test_id="BaseRunHolder", is_fcrd_down=True):
        self.test_id = test_id
        self.is_fcrd_down = is_fcrd_down
        self.time_zero = dt.datetime.utcnow()
        self.last_given_freq = 50.0
        self.test_start_time = None
        self.test_finish_time = None

    def SetTimeZero(self, time_zero):
        if type(time_zero) is float:
            time_zero = dt.datetime.fromtimestamp(time_zero)

        if type(time_zero) is not dt.datetime:
            raise Exception("Time zero must be either a float or a datetime object")

        self.time_zero = time_zero
        self.test_start_time = time_zero
        self.test_finish_time = time_zero


    # Calculate the response of the asset to a given timestamp
    # This assumes that the asset is initialized with a list of (time, _, freq) tuples
    # as well as a time zero
    def Input2Activation(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):
        raise NotImplementedError

    def IsOutsideSequence(self, **kwargs):
        raise NotImplementedError

    # Return last used frequency
    def LastTime2Frequency(self):
        return self.last_given_freq

    # Return last used test-id
    def LastTime2TestId(self):
        return self.test_id

    def TestStartTime(self):
        return self.test_start_time

    def TestFinishTime(self):
        return self.test_finish_time

    def SumOfTestActivations(self, fcrd_a):
        raise NotImplementedError

    def PredictedSoCChange(self, fcrd_a):
        raise NotImplementedError

class ChargeHolder(BaseRunHolder):
    def __init__(self, fcrd_a, test_id="Charging", soc_target_lower=0.0, soc_target_upper=1.0, fraction_of_max_power=0.3, measure_uncertainty=0.01):
        super().__init__(test_id=test_id)
        self.soc_target_lower = soc_target_lower
        self.soc_target_upper = soc_target_upper
        self.asset_in_nem = None
        self.at_target_soc = False
        self.at_bucketized_target_soc = False
        self.charging_activation_frequency = 50.1 + (fcrd_a.fcrddown_activation_saturation_f - 50.1) * fraction_of_max_power
        self.discharging_activation_frequency = 49.9 + (fcrd_a.fcrdup_activation_saturation_f - 49.9) * fraction_of_max_power
        self.measure_uncertainty = max(measure_uncertainty,0.001)
        self.no_soc_discretization_steps = round(1.0/measure_uncertainty) #100 # 100 s so we can have guarantee of 0.01 % soc, if granularity is 1% SOC
        self.no_soc_discretization_steps_inv = 1.0 / self.no_soc_discretization_steps
        self.saved_soc_permille_values = set()
        self.prev_soc_meas_perc = int(fcrd_a.GetSoC() * 10)
        self.prev_response = 0.0
        self.soc_est = fcrd_a.GetSoC() * 0.01
        self.bucket_size = None
        self.bucket_set_inspection_size = 1

        # We must double check that we have not set the soc target outside of non-NEM range,
        # i.e. there must be an overlap where we are achieving the target SoC, while not being in NEM
        # We have 6 different cases to consider:
        #   1) SOC-target subset of non-NEM range
        #       (do nothing - we will charge or discharge until we reach the target SoC)
        #   2) SOC-target superset of non-NEM range
        #       (do nothing - we will charge or discharge, until we reach the target SoC, then NEM activate until we are outside NEM range)
        #   3&4) SOCtarget overlap with non-NEM range
        #       (do nothing - we will charge or discharge, until we reach the target SoC, then NEM activate until we are outside NEM range))
        #   5&6) SOCtarget outside of non-NEM range (i.e. on one side)
        #       (adjust to being _just_ inside NEM range, otherwise we will charge/discharge to keep us outside non-NEM range)


        if self.soc_target_lower > self.soc_target_upper:
            raise Exception("Lower SoC target must be lower than upper SoC target when charging/recharging: " + str(self.soc_target_lower) + " > " + str(self.soc_target_upper))

        # This puts us into case 3 or 4
        if self.soc_target_lower > fcrd_a.SoC_disable_AEM_upperbound - measure_uncertainty: # some margin
            self.soc_target_lower = fcrd_a.SoC_disable_AEM_upperbound - measure_uncertainty

        if self.soc_target_upper < fcrd_a.SoC_disable_NEM_lowerbound + measure_uncertainty: # some margin
            self.soc_target_upper = fcrd_a.SoC_disable_NEM_lowerbound + measure_uncertainty  # some margin


    def IsBucketizedSOC(self):
        return len(self.saved_soc_permille_values) < self.no_soc_discretization_steps

    def GetSOCBucketSize(self):
        if not self.IsBucketizedSOC():
            self.bucket_size = None
            print("SOC is not bucketized, " + self.test_id)

        if self.bucket_set_inspection_size < len(self.saved_soc_permille_values):
            soc_buckets_list = list(self.saved_soc_permille_values)
            soc_buckets_list.sort()
            soc_diffs = [soc_buckets_list[i + 1] - soc_buckets_list[i] for i in range(len(soc_buckets_list)-1)]
            self.bucket_size = max(soc_diffs) * 0.001
            self.bucket_set_inspection_size = len(self.saved_soc_permille_values)
            print(f'self.bucket_size: {self.bucket_size}')
            print(f'self.bucket_set_inspection_size: {self.bucket_set_inspection_size}')
        else:
            self.bucket_size = None

        return self.bucket_size

    def AtTargetSoC(self, soc_meas):
        if self.IsBucketizedSOC():
            self.GetSOCBucketSize()
            if self.bucket_size is not None:
                self.at_bucketized_target_soc = self.soc_target_lower - self.bucket_size <= soc_meas <= self.soc_target_upper + self.bucket_size
                print(f'self.at_bucketized_target_soc: {self.at_bucketized_target_soc}')
                print(f'self.soc_target_lower - self.bucket_size: {self.soc_target_lower - self.bucket_size}')
                print(f'self.soc_target_upper + self.bucket_size: {self.soc_target_upper + self.bucket_size}')
                print(f'soc_meas: {soc_meas}')
                print(f'self.bucket_size: {self.bucket_size}')

                #return self.at_target_soc  # or self.at_target_smooth_soc


        self.at_target_soc = self.soc_target_lower - self.measure_uncertainty <= soc_meas <= self.soc_target_upper + self.measure_uncertainty

        return self.at_target_soc or self.at_bucketized_target_soc

    # Calculate the response of the asset given a soc target
    # This assumes that the asset is initialized with a list of (time, _, freq) tuples
    # as well as a time zero
    def Input2Activation(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):
        if soc_meas is None:
            soc_meas = fcrd_a.soc_estimate

        soc_meas_save = soc_meas

        # See if SOC "per mille" (part per thousand) can be cast to an integer
        # We use 0.1 of a per mille as a threshold
        # ToDo: generalize to non-integer bucket sizes
        sm_perc = soc_meas * 1000.0
        if math.ceil(10000 * soc_meas) - math.floor(10000 * soc_meas) < 1:
            #sm_perc = round(sm_perc)
            if sm_perc > 500: # Make sure we hit the NEM limits in sm_pers before in asset
                sm_perc = math.ceil(sm_perc)
            else:
                sm_perc = math.floor(sm_perc)

        self.saved_soc_permille_values.add(sm_perc)

        if self.IsBucketizedSOC():
            # The measurement changed "bucket"
            if not isinstance(self.prev_soc_meas_perc, type(sm_perc) or self.prev_soc_meas_perc != sm_perc):
                self.prev_soc_meas_perc = sm_perc
                # Trust the measurement, but don't discard the tracked estimate completely (90%-10%)
                self.soc_est = (0.9 * self.prev_soc_meas_perc * 0.001 + 0.1 * self.soc_est)
                #self.soc_est = self.prev_soc_meas_perc * 0.001
                print(f'self.soc_est: {self.soc_est}')
            else: # else we just update it one step
                self.soc_est = self.soc_est - fcrd_a.sampling_time * self.prev_response / fcrd_a.relative_capacity
        else:
            self.soc_est = soc_meas

        soc_meas = self.soc_est


        if self.test_start_time is None:
            self.test_start_time = time_meas

        self.asset_in_nem = fcrd_a.InNEM()

        if self.AtTargetSoC(soc_meas):
            given_freq = 50.0
            is_fcr_down = True
            self.test_finish_time = time_meas
        elif soc_meas <= self.soc_target_lower:
            given_freq = self.charging_activation_frequency
            is_fcr_down = True
        elif soc_meas >= self.soc_target_upper:
            given_freq = self.discharging_activation_frequency
            is_fcr_down = False
        else:
            raise Exception(
                "Should not be here. In Input2Activation SoC: " + str(soc_meas) + ", self.soc_target_lower: " + str(
                    self.soc_target_lower) + ", self.soc_target_upper: " + str(self.soc_target_upper))


        soc_meas = soc_meas_save
        self.last_given_freq = given_freq
        # Use the affine controller if we are in AEM
        if fcrd_a.IsAEMSoC(soc_meas):
            fcrd_a.UpdateEMTracking(given_freq, soc_meas=soc_meas, time_meas=time_meas)
            response = fcrd_a.Freq2ResponseFCRDAffine(given_freq)
            fcrd_a.UpdateSOC(response, soc_meas=soc_meas)
            response = response * 100
        elif is_fcr_down:
            #response = fcrd_a.Freq2ResponseLER(given_freq, soc_meas=soc_meas)
            response = fcrd_a.Freq2Response(given_freq, soc_meas=soc_meas, time_meas=time_meas)
        else:
            #response = fcrd_a.Freq2ResponseLERFcrDUp(given_freq, soc_meas=soc_meas)
            response = fcrd_a.Freq2ResponseFcrDUp(given_freq, soc_meas=soc_meas, time_meas=time_meas)

        self.prev_response = response * 0.01
        return response

    # We are finished if charging is done, and not in NEM
    def IsOutsideSequence(self, **kwargs):
        return (self.at_target_soc or self.at_bucketized_target_soc) and not self.asset_in_nem

    def PredictedSoCChange(self, fcrd_a):
        soc_est = fcrd_a.soc_estimate
        soc_diff = 0.0
        if soc_est < self.soc_target_lower:
            soc_diff = self.soc_target_lower - soc_est
        elif soc_est > self.soc_target_upper:
            soc_diff = self.soc_target_upper - soc_est

        print("Charge/discharge predicted SOC diff: " + str(soc_diff))

        return soc_diff


class SyncTimeHolder(ChargeHolder):
    def __init__(self, fcrd_a, test_id="SyncTimer", soc_target_lower=0.0, soc_target_upper=1.0, fraction_of_max_power=0.3, measure_uncertainty=0.01, reference_utc_time = None, synchronization_minute=0, synchronization_time=None):
        super().__init__(
            fcrd_a,
            test_id=test_id,
            soc_target_lower=soc_target_lower,
            soc_target_upper=soc_target_upper,
            fraction_of_max_power=fraction_of_max_power,
            measure_uncertainty=measure_uncertainty
        )

        if isinstance(synchronization_time, dt.datetime):
            self.synchronization_time = synchronization_time
        else:
            if reference_utc_time is None:
                reference_utc_time = dt.datetime.utcnow()

            if (reference_utc_time.tzinfo == None or reference_utc_time.tzinfo.utcoffset(reference_utc_time) == None):
                reference_utc_time = reference_utc_time.replace(tzinfo=dt.timezone.utc)

            synchronization_minute = int(synchronization_minute)

            # If we are past the synchronization minute in _this_ hour,
            # we must add 60 minutes s.t. we get to _next_ hour
            if reference_utc_time.minute > synchronization_minute:
                synchronization_minute = 60 + synchronization_minute

            self.synchronization_time = reference_utc_time.replace(microsecond=1, second=0, minute=0) + dt.timedelta(minutes=synchronization_minute)

    # We are finished if Timer is running out
    # Before this we should have run a ChargeHolder, and
    # the inherited Input2Activation( .. ) should have kept that SOC.
    # Therefore this method _only_ checks if the timer has run out
    def IsOutsideSequence(self, time_meas=None):
        if time_meas is None:
            time_meas = dt.datetime.utcnow()

        return time_meas >= self.synchronization_time


class TestHolder(BaseRunHolder):
    def __init__(self, test_id="Sequence test", is_fcrd_down=True):
        super().__init__(test_id=test_id, is_fcrd_down=is_fcrd_down)
        self.timed_frequency_sequence = []
        self.test_sequence_sampling_time = None
        self.current_tas_index = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def SetTimedFrequencySequence(self, timed_frequency_sequence):
        self.timed_frequency_sequence = timed_frequency_sequence
        self.test_sequence_sampling_time = timed_frequency_sequence[1][0] - timed_frequency_sequence[0][0]

    def SetTimeZero(self, time_zero):
        super().SetTimeZero(time_zero)
        self.timed_frequency_sequence = [(self.time_zero + dt.timedelta(seconds=t), *f_) for (t, *f_) in
                                         self.timed_frequency_sequence]
        self.current_tas_index = 0
        self.test_finish_time = self.timed_frequency_sequence[-1][0]

    # Calculate the response of the asset to a given timestamp
    # This assumes that the asset is initialized with a list of (time, _, freq) tuples
    # as well as a time zero
    def Input2Activation(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):

        if self.IsOutsideSequence():
            return 0.0

        while self.timed_frequency_sequence[self.current_tas_index][0] < time_meas:
            self.current_tas_index += 1
            if self.IsOutsideSequence():
                return 0.0

        given_freq = self.timed_frequency_sequence[self.current_tas_index][1]
        self.last_given_freq = given_freq
        if is_fcr_down:
            #return fcrd_a.Freq2ResponseLER(given_freq, soc_meas=soc_meas)
            return fcrd_a.Freq2Response(given_freq, soc_meas=soc_meas, time_meas=time_meas)
        else:
            #return fcrd_a.Freq2ResponseLERFcrDUp(given_freq, soc_meas=soc_meas)
            return fcrd_a.Freq2ResponseFcrDUp(given_freq, soc_meas=soc_meas, time_meas=time_meas)

    def IsOutsideSequence(self, **kwargs):
        return self.current_tas_index >= len(self.timed_frequency_sequence)

    # Return last used test-id
    def LastTime2TestId(self):
        in_index = self.current_tas_index
        if self.current_tas_index >= len(self.timed_frequency_sequence):
            in_index = len(self.timed_frequency_sequence) - 1

        # If we have no id, return "No test id"
        if len(self.timed_frequency_sequence[in_index]) < 3:
            return "no_test_id"

        return self.timed_frequency_sequence[in_index][2]

    def SumOfTestActivations(self, fcrd_a):

        if self.is_fcrd_down:
            fcrd_enable_freq = 50.1
            sum_activation = -sum([fcrd_a.Freq2ResponseFCRDAffine(f)  for _, f, _ in self.timed_frequency_sequence if f > fcrd_enable_freq])
        else:
            fcrd_enable_freq = 49.9
            sum_activation = -sum([fcrd_a.Freq2ResponseFCRDAffine(f)  for _, f, _ in self.timed_frequency_sequence if f < fcrd_enable_freq])

        return sum_activation * self.test_sequence_sampling_time

    def PredictedSoCChange(self, fcrd_a):
        sum_of_test_activations = self.SumOfTestActivations(fcrd_a)
        soc_diff = sum_of_test_activations / fcrd_a.relative_capacity
        return soc_diff


class RunFCRDHolder(BaseRunHolder):
    def __init__(self, test_id="one_hour_fcrd", reference_utc_time = None, runtime_mins=61):
        super().__init__(test_id=test_id, is_fcrd_down=True)

        if reference_utc_time is None:
            reference_utc_time = dt.datetime.utcnow()

        if (reference_utc_time.tzinfo == None or reference_utc_time.tzinfo.utcoffset(reference_utc_time) == None):
            reference_utc_time = reference_utc_time.replace(tzinfo=dt.timezone.utc)

        if not isinstance(runtime_mins, int) or runtime_mins < 1:
            runtime_mins = 61

        self.synchronization_time = reference_utc_time + dt.timedelta(minutes=runtime_mins)

    # Calculate the response of the asset for a given grid frequency
    # i.e. normal operation
    def Input2Activation(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):
        
        if not isinstance(time_meas, dt.datetime):
            # initialize with time now
            time_meas = dt.datetime.utcnow()
            time_meas = time_meas.replace(tzinfo=dt.timezone.utc)

        if not isinstance(self.test_start_time, dt.datetime):
            self.test_start_time = time_meas

        self.test_finish_time = time_meas
        self.last_given_freq = freq_meas
        if freq_meas >= fcrd_a.f0:
            #return fcrd_a.Freq2ResponseLER(freq_meas, soc_meas=soc_meas)
            return fcrd_a.Freq2Response(freq_meas, soc_meas=soc_meas, time_meas=time_meas)
        else:
            #return fcrd_a.Freq2ResponseLERFcrDUp(freq_meas, soc_meas=soc_meas)
            return fcrd_a.Freq2ResponseFcrDUp(freq_meas, soc_meas=soc_meas, time_meas=time_meas)

    def PredictedSoCChange(self, fcrd_a):
        return 0.0

    # We are finished if Timer is running out
    def IsOutsideSequence(self, time_meas=None):
        if time_meas is None:
            time_meas = dt.datetime.utcnow()

        return time_meas >= self.synchronization_time



class EMTestHolder(BaseRunHolder):# TestHolder):
    def __init__(self, test_id=None, is_fcrd_down=False):
        self.no_plan_steps = 5

        test_id = "fcrdup_em"
        if is_fcrd_down:
            test_id = "fcrddown_em"

        super().__init__(test_id=test_id, is_fcrd_down=is_fcrd_down)
        self.time_zero = None
        self.state_value = 0.0
        self.timer_sec = None # dt.timedelta(seconds=30.0)


    def InState(self, eval_value):
        return abs(self.state_value - eval_value) < 0.1

    def TimerFinished(self, time_meas):
        return time_meas - self.time_zero > self.timer_sec

    def IsOutsideSequence(self, **kwargs):
        return self.state_value >= 4.9 or self.state_value < 0.0

    # We do not know in advance how long the test will take, so we return 0.0
    def PredictedSoCChange(self, fcrd_a):
        print("In EMTestHolder.PredictedSoCChange" + str(self.test_id))
        return None

    # This method implements the state machine for the EM tests
    # The object has the state, and this method checks if it should remain, or change
    # State is a float, when it should have been an int or enum, but this way we follow the steps
    # in the ENTSOE document. It has integer steps, but has sub-steps within some integer steps.
    def StateMachine(self, fcrd_a, time_meas, soc_meas=None):
        if not isinstance(self.state_value, float) or self.state_value < 0.0:
            self.state_value = 0.0

        if self.time_zero is None:
            self.time_zero = time_meas

        if self.test_start_time is None:
            self.test_start_time = time_meas

        if self.timer_sec is None:
            self.timer_sec = dt.timedelta(seconds=30.0)

        # In first step, non-activation (50.09) for 30 sec or until NEM fades
        if self.InState(0):
            if self.TimerFinished(time_meas) and not fcrd_a.InNEM():
                self.state_value = 1.0
                self.time_zero = time_meas
                self.timer_sec = dt.timedelta(seconds=600.0)

        elif self.InState(1): #These steps (1 & 1.5) must be held until:
            # (1) NEM turns on when going into normal frequency band
            # (1.5) stay minimum 10 minutes (+ 60 sec buffert)
            if fcrd_a.NEMAllowed(50.0, soc_meas):
                self.state_value = 1.5
                # will enter NEM, wait at least 60 sec extra
                if self.TimerFinished(time_meas + dt.timedelta(seconds=60.0)):
                    self.time_zero = time_meas
                    self.timer_sec = dt.timedelta(seconds=60.0)

        elif self.InState(1.5):
            if self.TimerFinished(time_meas):
                self.state_value = 2.0
                self.time_zero = time_meas
                self.timer_sec = dt.timedelta(seconds=2.5*60.0)

        elif self.InState(2):
            # NEM turns on due to entering of normal frequency band (min 2,5 min)
            if self.TimerFinished(time_meas):
                self.state_value = 3.0
                self.time_zero = time_meas
                self.timer_sec = dt.timedelta(seconds=15*60.0)

        elif self.InState(3): # This step must be held 5 min after AEM turns on
            # (3) hold until AEM turns on
            # (3.5) hold for minimum 5 minutes (+ 60 sec buffer)
            if fcrd_a.IsAEMSoC(soc_meas):
                self.state_value = 3.5
                # in AEM, now wait at least 5 minutes (+ 60 sec buffer)
                if self.TimerFinished(time_meas + dt.timedelta(seconds=(5+1)*60.0)):
                    self.time_zero = time_meas
                    self.timer_sec = dt.timedelta(seconds= (5+1)*60.0)

        elif self.InState(3.5):
            # (3.5) hold for minimum 5 minutes (+ 60 sec buffer)
            if self.TimerFinished(time_meas):
                # Should not be needed - if we are in AEM we _will_ enter NEM, but just in case..
                if not fcrd_a.NEMAllowed(50.0, soc_meas):
                    pass # Not ready to leave state 3.5 yet..
                else:
                    self.state_value = 4.0
                    self.time_zero = time_meas
                    self.timer_sec = dt.timedelta(seconds=15.0*60.0)

        elif self.InState(4):
            # NEM must be turned on when stepping into the normal frequency band.
            # The step must be held until NEM and AEM turns off (min 15 min)
            if fcrd_a.InNEM():
                pass
            elif fcrd_a.InAEM():
                pass
            else:
                self.time_zero = time_meas
                self.timer_sec = dt.timedelta(seconds=(5 * 60.0))
                self.state_value = 4.5

        elif self.InState(4.5):
            if self.TimerFinished(time_meas):
                self.state_value = 5.0
                self.test_finish_time = time_meas

        else:
            self.state_value = 5.0 # We are done with the test

    # Calculate the diff of the frequency of the test, given an asset, timestamp and SoC
    def Input2FreqDiff(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):
        if soc_meas is None:
            soc_meas = fcrd_a.soc_estimate

        self.StateMachine(fcrd_a, time_meas, soc_meas=soc_meas)

        given_freq = 0.0
        if self.InState(0):
            given_freq = 0.09
        elif self.InState(1) or self.InState(1.5):
            given_freq = 0.5
        elif self.InState(2.0):
            given_freq = 0.09
        elif self.InState(3.0) or self.InState(3.5):
            given_freq = 0.5
        elif self.InState(4.0) or self.InState(4.5):
            given_freq = 0.09
        else:
            given_freq = 0.0

        return given_freq

    # Leaving this method to be implemented by child classes
    def Input2Activation(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):
        raise NotImplementedError



class FcrdDownEMTestHolder(EMTestHolder):
    def __init__(self):
        test_id = "fcrddown_em"
        is_fcrd_down = True
        EMTestHolder.__init__(self, test_id=test_id, is_fcrd_down=is_fcrd_down)

    def Input2Activation(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):
        fdiff = self.Input2FreqDiff(fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0)

        given_freq = 50.0 + fdiff
        self.last_given_freq = given_freq
        #return fcrd_a.Freq2ResponseLER(given_freq, soc_meas=soc_meas)
        return fcrd_a.Freq2Response(given_freq, soc_meas=soc_meas, time_meas=time_meas)

class FcrdUpEMTestHolder(EMTestHolder):
    def __init__(self):
        test_id = "fcrdup_em"
        is_fcrd_down = False
        EMTestHolder.__init__(self, test_id=test_id, is_fcrd_down=is_fcrd_down)

    def Input2Activation(self, fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0):
        fdiff = self.Input2FreqDiff(fcrd_a, time_meas, soc_meas=None, is_fcr_down=True, freq_meas=50.0)

        given_freq = 50.0 - fdiff
        self.last_given_freq = given_freq
        #return fcrd_a.Freq2ResponseLERFcrDUp(given_freq, soc_meas=soc_meas)
        return fcrd_a.Freq2ResponseFcrDUp(given_freq, soc_meas=soc_meas, time_meas=time_meas)


# Class for holding data regarding FCR-D PQ test(s).
# This is the interface to map input time to get asset activation responses
# This class does not hold any data regarding the asset itself, has any logic
# or algorithms how to control the asset, but rather on how to run the tests.

class FcrDTestSession:
    def __init__(
            self,
            asset_capacity: float = None,  # MWh
            asset_power: float = None,  # MW
            asset_soc: float = None, #0.5,  # initial SoC (in %)
            dbworker = None,
            tempo: float = 1.0,
            recharge_power: float = 30.0,
            verbose: bool = False,
            metadata_log_filename: str = 'metadata_log.csv',
            SoC_buffert_bound_in_minutes = 6,
            synchronization_minute = None,
            sampling_time = 0.1,
            test_sequence_sampling_time=0.1,
    ):
        self.is_ler = True
        if (asset_capacity is None or asset_power is None):
            raise Exception("asset_capacity and asset_power must be set")

        self.fcrd_asset = FcrDDownAsset(
            soc_estimate=asset_soc,  # unitless (max charge MWh / charge left MWh)
            absolute_capacity=asset_capacity,
            absolute_max_activation=asset_power,
            SoC_buffert_bound_in_minutes = SoC_buffert_bound_in_minutes,
            sampling_time = sampling_time
        )  # unit: h (max charge MWh / rated (i.e. max) discharge power MW)

        self.verbose = verbose
        self.synchronization_minute = synchronization_minute
        self.tempo = tempo  # 1 is default, lower is faster for faster debugging on HW
        self.test_list = []
        self.test_sequence_sampling_time = test_sequence_sampling_time  # This is the sampling time for the test sequence
        self.fcrdup_energy_management_start_soc = max(40.0, math.ceil(self.AssetRemainingSoC2EnergyManagementFcrDUp(0.0) * 100))
        self.fcrddown_energy_management_start_soc = 100.0 - self.fcrdup_energy_management_start_soc
        self.InitTests()
        self.run_on_CM10 = dbworker is not None
        self.dbworker = dbworker
        self.recharge_power = recharge_power
        self.test_metadata = []
        self.run_state = RunState.without_running_state
        self.soc_at_test_start = self.GetSoC()
        self.current_test_name = None
        self.metadata_log_filename = metadata_log_filename
        self.last_manual_frequency = 0.0
        self.contstatus_fcrdup = False
        self.contstatus_fcrddo = False
        self.current_test = None
        self.test_holder = None


    def InitTestMetadata(self, name, is_fcrd_down):

        new_dict = dict()
        new_dict['name'] = name
        new_dict['asset_soc_start'] = self.GetSoC()
        new_dict['is_fcrd_down'] = is_fcrd_down
        self.test_metadata.append(new_dict)

    def AppendTestMetadata(self):
        self.test_metadata[-1]['asset_soc_end'] = self.GetSoC()
        self.test_metadata[-1]['delta_soc'] = self.test_metadata[-1]['asset_soc_end'] - self.test_metadata[-1]['asset_soc_start']
        self.test_metadata[-1]['delta_energy (MWh)'] = self.test_metadata[-1]['delta_soc'] * 0.01 * self.fcrd_asset.absolute_capacity / (60 * 60)
        self.test_metadata[-1]['tot_time2 (s)'] = (self.test_holder.TestFinishTime() - self.test_holder.TestStartTime()).total_seconds()


    def IsTestFinished(self, **kwargs):
        return self.test_holder.IsOutsideSequence(**kwargs)


    # Print the metadata for the tests
    def WriteTestMetadata(self):
        keys = self.test_metadata[0].keys()
        top_row = ""
        for id in keys:
            top_row = top_row + ";" + id
        top_row = top_row + "\n"

        with open(self.metadata_log_filename, 'w') as f:
            f.write(top_row)
            for row in self.test_metadata:
                line = ""
                for id in keys:
                    line = line + ";" + str(row[id])
                line = line + "\n"
                f.write(line)


    def fcrddown_ramp_test(self, tempo, now_dt=None, **kwargs):
        endurance = 300  # or 900 for testing nonLER
        at_least_300 = 300
        list_of_ramps = [
            (30.0 * tempo, 50.1, 0.0),
            (4.9 * tempo, 50.55, 0.14),
            (55.1 * tempo, 50.1, 0.09),
            (endurance * tempo, 50.5, 0.24),
            (at_least_300 * tempo, 50.1, 0.24),
            (60 * tempo, 51.0, 0.24),
            (300 * tempo, 50.0, 0.24)
        ]

        list_of_ramps_ext = SeqOfPoint2SeqOfRamps(list_of_ramps)
        tot_seq = SequenceOfRampsBuilder(list_of_ramps_ext)

        tot_seq = [(t, f, "fcrddown_ramp") for t, f in tot_seq]
        # print(tot_seq)

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False

        self.test_holder = TestHolder(is_fcrd_down=True)

        self.test_holder.SetTimedFrequencySequence(tot_seq)
        self.test_holder.SetTimeZero(now_dt)

        return now_dt

    def fcrdup_static_linearity_test(self, tempo, now_dt=None, **kwargs):
        grid_freqs = [49.9, 49.8, 49.7, 49.6, 49.5, 49.6, 49.7, 49.8, 49.91]

        list_of_ramps = [(120.0 * tempo, gf, 0.14) for gf in grid_freqs]
        list_of_ramps[0] = (list_of_ramps[0][0], list_of_ramps[0][1], 0.0 )

        list_of_ramps_ext = SeqOfPoint2SeqOfRamps(list_of_ramps)
        tot_seq = SequenceOfRampsBuilder(list_of_ramps_ext, delta_x=self.test_sequence_sampling_time)
        tot_seq = [(t, f, "fcrdup_static_linearity") for t, f in tot_seq]

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False

        self.test_holder = TestHolder(is_fcrd_down=False)
        self.test_holder.SetTimedFrequencySequence(tot_seq)
        self.test_holder.SetTimeZero(now_dt)

        return now_dt


    def fcrdup_ramp_test(self, tempo, now_dt=None, **kwargs):
        endurance = 300  # or 900 for testing nonLER
        at_least_300 = 300
        list_of_ramps = [
            (30.0 * tempo, 50.0 - 0.1, 0.0),
            (4.9 * tempo, 50.0 - 0.55, 0.14),
            (55.1 * tempo, 50.0 - 0.1, 0.09),
            (endurance * tempo, 50.0 - 0.5, 0.24),
            (at_least_300 * tempo, 50.0 - 0.1, 0.24),
            (60 * tempo, 50.0 - 1.0, 0.24),
            (300 * tempo, 50.0, 0.24)
        ]

        list_of_ramps_ext = SeqOfPoint2SeqOfRamps(list_of_ramps)
        tot_seq = SequenceOfRampsBuilder(list_of_ramps_ext, delta_x=self.test_sequence_sampling_time)

        tot_seq = [(t, f, "fcrdup_ramp") for t, f in tot_seq]

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False

        self.test_holder = TestHolder(is_fcrd_down=False)

        self.test_holder.SetTimedFrequencySequence(tot_seq)
        self.test_holder.SetTimeZero(now_dt)

        return now_dt

    def SinefollowingSubTestSettings(self, baseline_offset = 50.3):

        sinefollowing_sub_test_settings = [
            (10, 5, 20, 300),  # last value is "suggested time for tests.. but no use for us..
            (15, 5, 15, 300),
            (25, 5, 10, 300),
            (40, 5, 7, 300),
            (50, 5, 7, 600),
            (60, 5, 7, 600),
            (70, 5, 7, 600)
        ]

        return sinefollowing_sub_test_settings, baseline_offset


    def fcrd_template_sine_test(self, tempo, now_dt=None, **kwargs):
        # Sine test
        sine_curve_scaling = 0.1

        baseline_offset = 50.3
        bo_key = "baseline_offset"
        if bo_key in kwargs.keys():
            baseline_offset = kwargs[bo_key]

        is_fcrd_down = True
        if baseline_offset < self.fcrd_asset.f0:
            is_fcrd_down = False

        sinefollowing_sub_test_settings, baseline_offset = self.SinefollowingSubTestSettings(baseline_offset=baseline_offset)

        import random
        tot_seq = []
        for sub_test_setting in sinefollowing_sub_test_settings:
            sine_period, no_stationary_periods, recomended_meas_periods, _ = sub_test_setting
            no_periods = recomended_meas_periods

            # sine_period, no_periods, baseline_frequency_signal = 50.3, sampling_time = 0.1
            tot_seq_sub = SequenceOfFrequencySignalsFollowingSineCurve(sine_period=sine_period,
                                                                       no_periods=no_periods,
                                                                       baseline_offset=baseline_offset,
                                                                       sine_curve_scaling=sine_curve_scaling,
                                                                       sampling_time=self.test_sequence_sampling_time
                                                                       )

            test_name = "sin_period"+str(sine_period)
            if not is_fcrd_down:
                test_name = "fcrdup_" + test_name

            tot_seq_sub = [(t, f, test_name) for t, f in tot_seq_sub]
            tot_seq = MergeSequences(tot_seq, tot_seq_sub)

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False

        self.test_holder = TestHolder(is_fcrd_down=is_fcrd_down)

        self.test_holder.SetTimedFrequencySequence(tot_seq)
        self.test_holder.SetTimeZero(now_dt)

        return now_dt

    def fcrddown_sine_test(self, tempo, now_dt=None, **kwargs):
        return self.fcrd_template_sine_test(tempo, now_dt=now_dt, **kwargs)


    def fcrdup_sine_test(self, tempo, now_dt=None, **kwargs):
        kwargs["baseline_offset"] = 50.0 - 0.3
        return self.fcrd_template_sine_test(tempo, now_dt=now_dt, **kwargs)


    def fcrddown_energy_management_test_state_based(self, tempo, now_dt=None, **kwargs):

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False
        self.test_holder = FcrdDownEMTestHolder()

        return now_dt


    def fcrdup_energy_management_test_state_based(self, tempo, now_dt=None, **kwargs):

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        self.fcrd_asset.in_alert_state_management_mode = False
        self.test_holder = FcrdUpEMTestHolder()

        return now_dt

    # This method assumes symmetrical ramps, i.e. if FcrD Up starts at 40% SoC,
    # then the FcrD down needs to start at 60% SoC.
    def FlipFcrdUpRamps2FcrdDownRamps(self, list_of_ramps):
        list_of_ramps_flipped = []
        for ramp in list_of_ramps:
            (duration, up_freq, abs_inclination) = ramp
            down_freq = abs(up_freq - self.fcrd_asset.f0) + self.fcrd_asset.f0
            list_of_ramps_flipped.append((duration, down_freq, abs_inclination))

        return list_of_ramps_flipped


    def chargedischarge_state_based(self, tempo, now_dt=None, **kwargs):

        soc_target_lower = 0.0
        soc_target_upper = 1.0
        fraction_of_max_power = 0.3
        measure_uncertainty = 0.001
        if not isinstance(kwargs, dict):
            kwargs = dict()

        if 'soc_target_lower' not in kwargs:
            kwargs['soc_target_lower'] = soc_target_lower

        if 'soc_target_upper' not in kwargs:
            kwargs['soc_target_upper'] = soc_target_upper

        if 'fraction_of_max_power' not in kwargs:
            kwargs['fraction_of_max_power'] = fraction_of_max_power

        if 'measure_uncertainty' not in kwargs:
            kwargs['measure_uncertainty'] = measure_uncertainty

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False
        self.test_holder = ChargeHolder(
            self.fcrd_asset,
            **kwargs
        )

        return now_dt

    def syncronized_start(self, tempo, now_dt=None, **kwargs):

        soc_target_lower = 0.0
        soc_target_upper = 1.0
        fraction_of_max_power = 0.3
        measure_uncertainty = 0.01
        synchronization_minute = 0 # Defaults to a whole hour
        synchronization_time = None
        if not isinstance(kwargs, dict):
            kwargs = dict()

        if 'soc_target_lower' not in kwargs:
            kwargs['soc_target_lower'] = soc_target_lower

        if 'soc_target_upper' not in kwargs:
            kwargs['soc_target_upper'] = soc_target_upper

        if 'fraction_of_max_power' not in kwargs:
            kwargs['fraction_of_max_power'] = fraction_of_max_power

        if 'measure_uncertainty' not in kwargs:
            kwargs['measure_uncertainty'] = measure_uncertainty

        if 'synchronization_minute' not in kwargs:
            kwargs['synchronization_minute'] = synchronization_minute

        if 'synchronization_time' not in kwargs:
            kwargs['synchronization_time'] = synchronization_time
        elif not isinstance(kwargs['synchronization_time'], dt.datetime):
            print("synchronization_time must be a datetime object, synchronization time will be set using other parameters")
            time.sleep(1.0)

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        if 'reference_utc_time' not in kwargs:
            kwargs['reference_utc_time'] = now_dt



        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False
        self.test_holder = SyncTimeHolder(
            self.fcrd_asset,
            **kwargs
        )

        return now_dt

    def one_hour_activation_run(self, tempo, now_dt=None, **kwargs):
        runtime_mins = 61

        kwargs = dict()

        if now_dt is None:
            now_dt = dt.datetime.fromtimestamp(
                time.time()
            )

        if 'reference_utc_time' not in kwargs:
            kwargs['reference_utc_time'] = now_dt

        if 'runtime_mins' not in kwargs:
            kwargs['runtime_mins'] = runtime_mins


        # If we happen to be in AEM, on our way out, lets _set_ it to out
        # (it will be set to True in asset class if we _are_ in AEM)
        self.fcrd_asset.in_alert_state_management_mode = False
        self.test_holder = RunFCRDHolder(
            **kwargs
        )

        return now_dt

    def CheckBounds(self, meas_uncertainty=0.02):
        # Testing each test bounds, raising error if not possible

        if self.fcrd_asset.SoC_disable_NEM_lowerbound + meas_uncertainty > self.fcrd_asset.SoC_enable_NEM_upperbound:
            raise ValueError("The NEM disable lowerbound is higher than the AEM enable upperbound (+ measure uncertainty): "
                             + str(self.fcrd_asset.SoC_disable_NEM_lowerbound) + " + " + str(meas_uncertainty) + " > "
                             + str(self.fcrd_asset.SoC_enable_NEM_upperbound))

        self.fcrddown_ramp_test(self.tempo)
        est_soc = self.test_holder.PredictedSoCChange(self.fcrd_asset)
        if self.fcrd_asset.SoC_disable_NEM_lowerbound + est_soc > self.fcrd_asset.SoC_enable_AEM_upperbound:
            print(f"test soc change: {est_soc}")
            print(f"self.fcrd_asset.SoC_enable_AEM_upperbound: {self.fcrd_asset.SoC_enable_NEM_upperbound}")
            print(f"self.fcrd_asset.SoC_disable_NEM_lowerbound: {self.fcrd_asset.SoC_disable_NEM_lowerbound}")
            raise ValueError("The ramp test is not possible to run, since the SoC will enter AEM during sin tests (consider lowering the power")

        self.fcrddown_sine_test(self.tempo)
        est_soc = self.test_holder.PredictedSoCChange(self.fcrd_asset)
        if self.fcrd_asset.SoC_disable_NEM_lowerbound + est_soc > self.fcrd_asset.SoC_enable_AEM_upperbound:
            print(f"test soc change: {est_soc}")
            print(f"self.fcrd_asset.SoC_enable_AEM_upperbound: {self.fcrd_asset.SoC_enable_AEM_upperbound}")
            print(f"self.fcrd_asset.SoC_disable_NEM_lowerbound: {self.fcrd_asset.SoC_disable_NEM_lowerbound}")
            raise ValueError("The sine test is not possible to run, since the SoC will enter AEM during sin tests (consider splitting test into sub-tests   ")


    def InitTests(self, meas_uncertainty=0.001):
        # List of tuples - method, is_fcr_down, min soc, max soc, soc-"cost"(?)
        self.CheckBounds(meas_uncertainty=meas_uncertainty)
        self.PrintParameters()

        # Bounds for starting sin tests
        nem_disable_lb = self.fcrd_asset.SoC_disable_NEM_lowerbound

        sin_est_soc = self.test_holder.PredictedSoCChange(self.fcrd_asset)
        sin_lb = nem_disable_lb
        sin_ub = self.fcrd_asset.SoC_enable_AEM_upperbound - sin_est_soc #- meas_uncertainty

        # Bounds for starting (fcrd down) ramp tests
        self.fcrddown_ramp_test(self.tempo)
        frcddown_ramp_est_soc = self.test_holder.PredictedSoCChange(self.fcrd_asset)
        ramp_fcrddown_lb = self.fcrd_asset.SoC_enable_NEM_lowerbound
        ramp_fcrddown_ub = self.fcrd_asset.SoC_enable_NEM_upperbound - frcddown_ramp_est_soc #- meas_uncertainty

        # Bounds for starting (fcrd up) ramp tests - assuming symmetry with fcrd down tests
        ramp_fcrdup_lb = 0.5 + (0.5 - ramp_fcrddown_ub)
        ramp_fcrdup_ub = 0.5 + (0.5 - ramp_fcrddown_lb)

        # Bounds for starting (fcrd up) FCRD up/down run
        nem_disable_ub = self.fcrd_asset.SoC_disable_NEM_upperbound
        mid_soc = 0.5

        em_fcrd_down_lb = 0.5
        em_fcrd_down_ub = nem_disable_ub

        em_fcrd_up_lb = nem_disable_lb
        em_fcrd_up_ub = 0.5

        # Print the limits
        if self.verbose:
            print(f"sin_lb: {sin_lb}")
            print(f"sin_ub: {sin_ub}")
            print(f"sin_est_soc ramp: {sin_est_soc}")

            print(f"ramp_fcrddown_lb: {ramp_fcrddown_lb}")
            print(f"ramp_fcrddown_ub: {ramp_fcrddown_ub}")

            print(f"ramp_fcrdup_lb: {ramp_fcrdup_lb}")
            print(f"ramp_fcrdup_ub: {ramp_fcrdup_ub}")

            print(f"nem_disable_lb: {nem_disable_lb}")
            print(f"frcddown_ramp_est_soc ramp: {frcddown_ramp_est_soc}")

            print(f"nem_disable_ub: {nem_disable_ub}")
            print(f"mid_soc: {mid_soc}")

        empty_kwargs = { "meas_uncertainty" : meas_uncertainty }
        self.test_list = [
            (self.fcrdup_energy_management_test_state_based, False, em_fcrd_down_lb, em_fcrd_down_ub, empty_kwargs),
            (self.fcrddown_ramp_test, True, ramp_fcrddown_lb, ramp_fcrddown_ub, empty_kwargs),
            (self.fcrdup_ramp_test, False, ramp_fcrdup_lb, ramp_fcrdup_ub, empty_kwargs),  # analogous to above
            (self.fcrddown_sine_test, True, sin_lb, sin_ub, empty_kwargs),
            (self.fcrddown_energy_management_test_state_based, True, em_fcrd_up_lb, em_fcrd_up_ub, empty_kwargs),
            (self.one_hour_activation_run, None, mid_soc, nem_disable_ub, empty_kwargs)
        ]

        # Exchange the charge-state bounds with a ChargeHolder object doing the same thing,
        # while avoiding NEM.
        tmp_list = []
        for index, item in enumerate(self.test_list):
            tmp_kwargs = {'soc_target_lower': item[2], 'soc_target_upper': item[3]}
            if "meas_uncertainty" in item[4]:
                tmp_kwargs['measure_uncertainty'] = item[4]['meas_uncertainty']
            chrg_tpl = (self.chargedischarge_state_based, None, 0.0, 100.0, tmp_kwargs)
            tmp_list.append(chrg_tpl)

            if index == 0 and isinstance(self.synchronization_minute, int):
                tmp_kwargs = {'soc_target_lower': item[2], 'soc_target_upper': item[3], 'synchronization_minute': self.synchronization_minute}
                chrg_tpl = (self.syncronized_start, None, 0.0, 100.0, tmp_kwargs)
                tmp_list.append(chrg_tpl)

            tmp_tpl = (item[0], item[1], 0.0, 1.0, item[4])
            tmp_list.append(tmp_tpl)

        self.test_list = tmp_list

        self.run_state = RunState.without_running_state

    def GetSoC(self):
        return self.fcrd_asset.GetSoC()

    def FractionalPowerNEM(self):
        return self.fcrd_asset.FractionalPowerNEM()

    def NEMCurrent(self):
        return self.fcrd_asset.NEMCurrent()

    def FreqRef(self):
        return self.fcrd_asset.FreqRef()

    def IsAEMSoC(self, soc_meas):
        return self.fcrd_asset.IsAEMSoC(soc_meas)

    def AssetPower(self):
        return self.fcrd_asset.C_fcrd

    def EnergyManagementSocLowerBound(self):
        # when self.AssetRemainingEnduranceFcrDDown(soc_meas) = self.fcrd_asset.SoC_enable_NEM_bound_in_minutes
        # i.e.
        # (1.0 - soc_meas) / self.fcrd_asset.SoC_bound_factor = self.fcrd_asset.SoC_enable_NEM_bound_in_minutes
        # 1.0 - soc_meas = self.fcrd_asset.SoC_enable_NEM_bound_in_minutes * self.fcrd_asset.SoC_bound_factor
        # soc_meas = 1.0 - self.fcrd_asset.SoC_enable_NEM_bound_in_minutes * self.fcrd_asset.SoC_bound_factor
        return 1.0 - self.fcrd_asset.SoC_enable_NEM_bound_in_minutes * self.fcrd_asset.SoC_bound_factor


    # Remaining endurance for FCR-D Down (until 100%, in minutes)
    def AssetRemainingEnduranceFcrDDown(self, soc_meas):
        return (1.0 - soc_meas) / self.fcrd_asset.SoC_bound_factor

    def AssetRemainingSoCEnduranceFcrDDown(self, time_in_minutes):

        # Given time_in_minutes = (1.0 - soc_meas) / self.fcrd_asset.SoC_bound_factor
        # we get soc_meas = 1.0 - time_in_minutes * self.fcrd_asset.SoC_bound_factor
        return 1.0 - time_in_minutes * self.fcrd_asset.SoC_bound_factor

    def AssetRemainingSoCEnduranceFcrDUp(self, time_in_minutes):

        # Given time_in_minutes = soc_meas / self.fcrd_asset.SoC_bound_factor
        # we get soc_meas = time_in_minutes * self.fcrd_asset.SoC_bound_factor
        return time_in_minutes * self.fcrd_asset.SoC_bound_factor


    # This gives remaining time until NEM* is enabled by Fcrd Down
    # * We look for NEM since it will be activated before AEM
    def AssetRemainingSoC2EnergyManagementFcrDDown(self, time_in_minutes):
        return self.AssetRemainingSoCEnduranceFcrDDown(time_in_minutes + self.fcrd_asset.SoC_enable_NEM_bound_in_minutes)

    # This gives remaining time until NEM* is enabled by Fcrd Up
    # * We look for NEM since it will be activated before AEM
    def AssetRemainingSoC2EnergyManagementFcrDUp(self, time_in_minutes):
        return self.AssetRemainingSoCEnduranceFcrDUp(time_in_minutes + self.fcrd_asset.SoC_enable_NEM_bound_in_minutes)


    # Remaining endurance for FCR-D Up (until 0%, in minutes)
    def AssetRemainingEnduranceFcrDUp(self, soc_meas):
        return soc_meas * 1.0 / self.fcrd_asset.SoC_bound_factor

    # Remaining endurance for FCR-D Down (until NEM, in minutes)
    def MinutesToAssetSoCEnableNEMBoundsFcrDDown(self, soc_meas):
        return self.AssetRemainingEnduranceFcrDDown(soc_meas) - self.fcrd_asset.SoC_enable_NEM_bound_in_minutes

    # Remaining endurance for FCR-D Down (until NEM, in minutes)
    def MinutesToAssetSoCEnableNEMBoundsFcrDUp(self, soc_meas):
        return self.AssetRemainingEnduranceFcrDUp(soc_meas) - self.fcrd_asset.SoC_enable_NEM_bound_in_minutes

    # Remaining endurance for FCR-D Down (until NEM, in minutes)
    def MinutesToAssetSoCEnableAEMBoundsFcrDUp(self, soc_meas):
        return self.AssetRemainingEnduranceFcrDUp(soc_meas) - self.fcrd_asset.SoC_enable_AEM_bound_in_minutes

    def TestFinished(self):
        return self.run_state is RunState.have_no_tests_to_run

    # Return last used frequency, if we are running tests.
    # Otherwise return nominal grid frequency (50.0 Hz),
    # Even if we are charging or discharging.
    def LastAppliedFrequencySignal(self):

        if self.run_state is RunState.running_subtest:
            return self.test_holder.LastTime2Frequency()
        elif self.run_state is RunState.running_charge_or_discharge:
            return self.fcrd_asset.f0
        elif self.run_state is RunState.without_running_state:
            return self.last_manual_frequency
        elif self.run_state is RunState.have_no_tests_to_run:
            return self.last_manual_frequency
        else:
            raise Exception("Unknown run state in LastAppliedFrequencySignal: " + str(self.run_state))

    # Return last used frequency, if we are running tests.
    # Otherwise return nominal grid frequency (50.0 Hz),
    # Even if we are charging or discharging.
    def LastAppliedTestId(self):

        if self.run_state is RunState.running_subtest:
            return self.test_holder.LastTime2TestId()
        elif self.run_state is RunState.running_charge_or_discharge:
            return "charge_or_discharge"
        elif self.run_state is RunState.without_running_state:
            return "no_state"
        elif self.run_state is RunState.have_no_tests_to_run:
            return "no_tests_to_run"
        else:
            raise Exception("Unknown run state in LastAppliedTestId: " + str(self.run_state))


    def EvalHaveNoTestsToRun(self, time_meas, soc_meas):
        self.contstatus_fcrdup = False
        self.contstatus_fcrddo = False

        if self.verbose:
            print("EvalHaveNoTestsToRun")

        # Should not be here.. but let's check anyway
        if len(self.test_list) > 0:
            self.run_state = RunState.without_running_state
            return self.EvalWithoutRunningState(time_meas, soc_meas)
        return time_meas, soc_meas

    # This state should always change - either we have nothing left to run,
    # or we get into a subtest, or we get into a charge/discharge state
    def EvalWithoutRunningState(self, time_meas, soc_meas):
        self.contstatus_fcrdup = False
        self.contstatus_fcrddo = False

        if self.verbose:
            print("EvalWithoutRunningState")
        if len(self.test_list) == 0:
            self.WriteTestMetadata()
            self.run_state = RunState.have_no_tests_to_run
            return time_meas, soc_meas

        seq_method, is_fcrd_down, soc_min, soc_max, run_object_kwargs = self.test_list[0]
        if not self.is_ler or soc_min < soc_meas < soc_max:
            self.contstatus_fcrdup = not is_fcrd_down
            self.contstatus_fcrddo = is_fcrd_down
            self.current_test = self.test_list[0]

            self.run_state = RunState.running_subtest
            # Need to be explained (and renamed):
            # This initiates next sequence in the list _to_ the frcd_asset object,
            # and returns the first time instance
            time_meas = seq_method(self.tempo, now_dt=time_meas, **run_object_kwargs)
            #self.test_holder.PredictedSoCChange(self.fcrd_asset)

            # Store data for logging
            self.InitTestMetadata(str(seq_method), is_fcrd_down=is_fcrd_down)
            # Change values for SvK logs
            self.test_list.pop(0)
            return time_meas, soc_meas
        else:
            self.run_state = RunState.running_charge_or_discharge
            return time_meas, soc_meas


    def EvalRunningSubtest(self, time_meas, soc_meas):
        if self.verbose:
            print("EvalRunningSubtest")
        if self.IsTestFinished(time_meas=time_meas):
            self.current_test = None
            self.AppendTestMetadata()
            self.run_state = RunState.without_running_state
            return self.EvalWithoutRunningState(time_meas, soc_meas)
        else:
            return time_meas, soc_meas

    # This state should change if we are finished charging/discharging
    # or continue if we are not finished
    def EvalRunningChargeOrDischarge(self, time_meas, soc_meas):
        self.contstatus_fcrdup = False
        self.contstatus_fcrddo = False

        if self.verbose:
            print("EvalRunningChargeOrDischarge")

        _, is_fcrd_down, soc_min, soc_max, _ = self.test_list[0]
        if not (soc_min < soc_meas < soc_max):
            return time_meas, soc_meas
        else:
            # Should not be here.. but let's check anyway
            self.run_state = RunState.without_running_state
            return self.EvalWithoutRunningState(time_meas, soc_meas)


    # This method will evaluate the current state and update the state machine
    def ComputeStateTransition(self, time_meas, soc_meas):
        if self.verbose:
            print("ComputeStateTransition")
        if self.run_state is RunState.have_no_tests_to_run:
            time_meas, soc_meas = self.EvalHaveNoTestsToRun(time_meas, soc_meas)
        elif self.run_state is RunState.without_running_state:
            time_meas, soc_meas = self.EvalWithoutRunningState(time_meas, soc_meas)
        elif self.run_state is RunState.running_subtest:
            time_meas, soc_meas = self.EvalRunningSubtest(time_meas, soc_meas)
        elif self.run_state is RunState.running_charge_or_discharge:
            time_meas, soc_meas = self.EvalRunningChargeOrDischarge(time_meas, soc_meas)
        else:
            raise Exception("Unknown run state in ComputeStateTransition: " + str(self.run_state))

        return time_meas, soc_meas

    # This method will write the current state to the database
    def WriteToDB(self, soc_meas):#, is_fcr_down):

        # Activated NEM power
        nempct = self.FractionalPowerNEM() * self.AssetPower()

        # Log the "in-AEM-state"
        aem = self.IsAEMSoC(soc_meas) if self.is_ler else False
        aem = 1 if aem else 0

        # Placeholder
        # (Thoughts after huddle 18 oct): either how you controling the contoller aem or nem or linear
        # performance or stability mode, static or dynamic. Checks with SvK. Leave for now.
        contmode_fcrddo = "FCRDUPDOWN1"
        if self.current_test is not None:
            _, is_fcr_down, _,_,_ = self.current_test
            contmode_fcrddo = "FCRDDOWN1" if is_fcr_down else "FCRDUP1"

        # ApplFreqSig - last applied frequency signal
        applfreqsig = self.LastAppliedFrequencySignal()

        # test_id - last applied test id, used for ease of separating data later
        test_id = self.LastAppliedTestId()

        # SOC - state of charge as estimated by the asset
        SOC = self.GetSoC()

        # Are we in FcrDUp mode
        # ContStatus_FcrdUp - contstatus_fcrdup

        # Are we in FcrDDown mode
        # ContStatus_FcrdDo - contstatus_fcrddo

        # What is asset setpoint (0.0 for batteries, runpoint for hydro etc))
        # ContSetP - self.fcrd_asset.contsetp

        # ResSize_FcrdUp - Remaining Endurance for FcrDUp (in mins)
        ressize_fcrdup = self.AssetRemainingEnduranceFcrDUp(soc_meas) if self.is_ler else "Inf"

        # ResSize_FcrdDo - Remaining Endurance for FcrDDown (in mins)
        ressize_fcrddo = self.AssetRemainingEnduranceFcrDDown(soc_meas)

        if self.verbose:
            print("nempct: " + str(nempct))
            print("AEM " + str(aem))
            print("ContMode_FcrdDo " + str(contmode_fcrddo))
            print("ApplFreqSig: " + str(applfreqsig))
            print("ContStatus_FcrdUp: " + str(self.contstatus_fcrdup))
            print("ContStatus_FcrdDo: " + str(self.contstatus_fcrddo))
            print("ContSetP: " + str(self.fcrd_asset.contsetp))
            print("ResSize_FcrdUp: " + str(ressize_fcrdup))
            print("ResSize_FcrdDo: " + str(ressize_fcrddo))
            print("test_id: " + str(test_id))
            print("SOC: " + str(SOC))

        if self.run_on_CM10:
            self.dbworker.queueFr(dt.datetime.utcnow(), nempct, "NEM")
            self.dbworker.queueFr(dt.datetime.utcnow(), aem, "AEM")
            self.dbworker.queueFr(dt.datetime.utcnow(), contmode_fcrddo, "ContMode_FcrdDo")
            self.dbworker.queueFr(dt.datetime.utcnow(), applfreqsig, "ApplFreqSig")
            self.dbworker.queueFr(dt.datetime.utcnow(), self.contstatus_fcrdup, "ContStatus_FcrdUp")
            self.dbworker.queueFr(dt.datetime.utcnow(), self.contstatus_fcrddo, "ContStatus_FcrdDo")
            self.dbworker.queueFr(dt.datetime.utcnow(), self.fcrd_asset.contsetp, "ContSetP")
            self.dbworker.queueFr(dt.datetime.utcnow(), ressize_fcrdup, "ResSize_FcrdUp")
            self.dbworker.queueFr(dt.datetime.utcnow(), ressize_fcrddo, "ResSize_FcrdDo")
            self.dbworker.queueFr(dt.datetime.utcnow(), test_id, "test_id")
            self.dbworker.queueFr(dt.datetime.utcnow(), SOC, "SOC")

    # This is the interface to map input time to get asset activation responses
    # It calls the implementation, and then writes to the database
    def TimeAndPower2Activation(self, time_meas, soc_meas=None, nominal_power=0.0):

        grid_freq = self.fcrd_asset.NominalPower2Freq(nominal_power * 0.01)
        res = self.Time2ActivationImpl(time_meas, soc_meas, grid_freq=grid_freq) #, is_fcr_down)

        if self.run_on_CM10:
            self.WriteToDB(soc_meas) #, is_fcr_down)

        return res


    # This is the interface to map input time to get asset activation responses
    # It calls the implementation, and then writes to the database
    def Time2Activation(self, time_meas, soc_meas=None, grid_freq=50.0): #, is_fcr_down=True):
        res = self.Time2ActivationImpl(time_meas, soc_meas, grid_freq=grid_freq) #, is_fcr_down)

        if self.run_on_CM10:
            self.WriteToDB(soc_meas) #, is_fcr_down)

        return res


    # This is the implementation of the interface to map input time to get asset activation responses
    # It keeps track on what to run next, regardless if it is a new experiment
    # next step on the current excperiment, or just stop activation (return zero).
    # This is the key abstraction interface for FCR-D tests. This holds the logic
    # on how to run the tests.
    # All the logic on how to compute the activation given state of the asset belongs to
    # the FcrDDownAsset class.
    def Time2ActivationImpl(self, time_meas, soc_meas=None, grid_freq=50.0):#, is_fcr_down=True):
        if self.verbose:
            print("Time2Activation")

        time_meas, soc_meas = self.ComputeStateTransition(time_meas, soc_meas)

        if self.verbose:
            print("Time2Activation - step2")

        # Start by checking if we should change state
        if self.run_state is RunState.have_no_tests_to_run:
            self.last_manual_frequency = self.fcrd_asset.f0
            return 0.0
        elif self.run_state is RunState.without_running_state:
            raise Exception("Should not be here")
        elif self.run_state is RunState.running_subtest:
            if self.verbose:
                print("Time2Activation - Subtest")

            _, is_fcr_down, _,_,_ = self.current_test
            self.contstatus_fcrdup = not is_fcr_down
            self.contstatus_fcrddo = is_fcr_down

            t2a2 = self.test_holder.Input2Activation(
                self.fcrd_asset,
                time_meas,
                soc_meas=soc_meas,
                is_fcr_down=is_fcr_down,
                freq_meas=grid_freq
            )

            return t2a2
        elif self.run_state is RunState.running_charge_or_discharge:
            return self.GetChargeDischargeResponse(soc_meas=soc_meas)
        else:
            raise Exception("Unknown run state in Time2Activation: " + str(self.run_state))

    # This method is used to get the charge/discharge response,
    # given the upper/lower requirements of entering the test
    def GetChargeDischargeResponse(self, soc_meas=None):
        if self.verbose:
            print("Time2Activation - Charge or discharge")
        _, _, soc_min, soc_max, _ = self.test_list[0]
        if soc_min < soc_meas:
            if self.verbose:
                print("Time2Activation - Charge")
            power_output_percent = self.recharge_power
        elif soc_meas < soc_max:
            if self.verbose:
                print("Time2Activation - Discharge")
            power_output_percent = self.recharge_power * -1.0
        else:
            raise Exception("Should not be here: charge/discharge while soc_min: " + str(soc_min) + "soc_max: " +
                            str(soc_max) + "soc_meas: " + str(soc_meas) + "next test: " +
                            str(None if len(self.test_list) < 1 else self.test_list[0]))

        self.fcrd_asset.UpdateSOC(power_output_percent * 0.01, soc_meas=soc_meas)

        return power_output_percent

    def Freq2ActivationLER(self, freq_meas, soc_meas=None):
        is_fcr_down = True
        if freq_meas < self.fcrd_asset.f0:
            is_fcr_down = False

        # If we change to this state, it will come back to the state machine if a test would resume.
        self.run_state = RunState.without_running_state
        self.last_manual_frequency = freq_meas
        self.contstatus_fcrdup = True
        self.contstatus_fcrddo = True

        if is_fcr_down:
            return self.fcrd_asset.Freq2ResponseLER(freq_meas, soc_meas=soc_meas)
        else:
            return self.fcrd_asset.Freq2ResponseLERFcrDUp(freq_meas, soc_meas=soc_meas)

    def PrintParameters(self):

        print("self.fcrd_asset.SoC_disable_AEM_lowerbound ", end="")
        print(self.fcrd_asset.SoC_disable_AEM_lowerbound)

        print("self.fcrd_asset.SoC_enable_AEM_lowerbound ", end="")
        print(self.fcrd_asset.SoC_enable_AEM_lowerbound)

        print("self.fcrd_asset.SoC_enable_NEM_upperbound ", end="")
        print(self.fcrd_asset.SoC_enable_NEM_upperbound)

        print("self.fcrd_asset.SoC_disable_NEM_upperbound ", end="")
        print(self.fcrd_asset.SoC_disable_NEM_upperbound)

        print("self.fcrd_asset.SoC_disable_NEM_lowerbound ", end="")
        print(self.fcrd_asset.SoC_disable_NEM_lowerbound)

        print("self.fcrd_asset.SoC_enable_NEM_lowerbound ", end="")
        print(self.fcrd_asset.SoC_enable_NEM_lowerbound)

        # Print these:
        # self.fcrd_asset.SoC_enable_AEM_bound_in_minutes
        # self.fcrd_asset.SoC_disable_AEM_bound_in_minutes
        # self.fcrd_asset.SoC_enable_NEM_bound_in_minutes
        # self.fcrd_asset.SoC_disable_NEM_bound_in_minutes
        print("self.fcrd_asset.SoC_enable_AEM_bound_in_minutes ", end="")
        print(self.fcrd_asset.SoC_enable_AEM_bound_in_minutes)
        print("self.fcrd_asset.SoC_disable_AEM_bound_in_minutes ", end="")
        print(self.fcrd_asset.SoC_disable_AEM_bound_in_minutes)
        print("self.fcrd_asset.SoC_enable_NEM_bound_in_minutes ", end="")
        print(self.fcrd_asset.SoC_enable_NEM_bound_in_minutes)
        print("self.fcrd_asset.SoC_disable_NEM_bound_in_minutes ", end="")
        print(self.fcrd_asset.SoC_disable_NEM_bound_in_minutes)

class FcrDTestSessionNonLER(FcrDTestSession):
    def __init__(
            self,
            asset_power: float = None,  # MW
            dbworker = None,
            tempo: float = 1.0,
            verbose: bool = False,
            metadata_log_filename: str = 'metadata_log.csv',
            synchronization_minute = None,
            test_sequence_sampling_time=0.1
    ):
        self.test_sequence_sampling_time = test_sequence_sampling_time  # This is the sampling time for the test sequence

        self.is_ler = False

        if not isinstance(asset_power, float):
            asset_power = 1.0

        self.fcrd_asset = FcrDAssetNoEnergyReservoir(
            absolute_max_activation=asset_power
        )

        self.verbose = verbose
        self.synchronization_minute = synchronization_minute
        self.tempo = tempo  # 1 is default, lower is faster for faster debugging on HW
        self.test_list = []
        self.InitTests()
        self.run_on_CM10 = dbworker is not None
        self.dbworker = dbworker
        self.test_metadata = []
        self.run_state = RunState.without_running_state
        self.current_test_name = None
        self.metadata_log_filename = metadata_log_filename
        self.last_manual_frequency = 0.0
        self.contstatus_fcrdup = False
        self.contstatus_fcrddo = False
        self.current_test = None
        self.test_holder = None

    def CheckBounds(self, meas_uncertainty=0.02):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    def InitTests(self, meas_uncertainty=0.001):
        # List of tuples - method, is_fcr_down, min soc, max soc, soc-"cost"(?)


        empty_kwargs = dict()
        self.test_list = [
            #(self.fcrddown_ramp_test, True, None, None, empty_kwargs),
            (self.fcrdup_ramp_test, False, None, None, empty_kwargs),  # analogous to above
            (self.fcrdup_sine_test, False, None, None, empty_kwargs),
            #(self.fcrddown_sine_test, True, None, None, empty_kwargs),
            (self.one_hour_activation_run, None, None, None, empty_kwargs)
        ]

        # Possibly add synchronization
        if isinstance(self.synchronization_minute, int):
            tmp_kwargs = {'synchronization_minute': self.synchronization_minute}
            chrg_tpl = (self.syncronized_start, None, 0.0, 100.0, tmp_kwargs)
            self.test_list.insert(0, chrg_tpl)

        self.run_state = RunState.without_running_state

    # This state should change if we are finished charging/discharging
    # or continue if we are not finished
    def EvalRunningChargeOrDischarge(self, time_meas, soc_meas):
        self.contstatus_fcrdup = False
        self.contstatus_fcrddo = False

        self.run_state = RunState.without_running_state
        print("Here!")
        return self.EvalWithoutRunningState(time_meas, soc_meas)

    def GetChargeDischargeResponse(self, soc_meas=None):
        print("Time2Activation - Zero charge for Charge/Discharge with Non-LER asset.")
        return 0.0


    def AppendTestMetadata(self):
        self.test_metadata[-1]['tot_time2 (s)'] = (self.test_holder.TestFinishTime() - self.test_holder.TestStartTime()).total_seconds()


    def GetSoC(self):
        return 50.0 # not ideal, but have to work for now (since I know there are usages of this method)

    def FractionalPowerNEM(self):
        return 0.01 # not ideal, but have to work for now (since I know there are usages of this method)

    def NEMCurrent(self):
        #return self.fcrd_asset.NEMCurrent()
        raise NotImplementedError("This method is not implemented for non-LER tests")
    def FreqRef(self):
        #return self.fcrd_asset.FreqRef()
        raise NotImplementedError("This method is not implemented for non-LER tests")

    def IsAEMSoC(self, soc_meas):
        #return self.fcrd_asset.IsAEMSoC(soc_meas)
        raise NotImplementedError("This method is not implemented for non-LER tests")

    def EnergyManagementSocLowerBound(self):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    # Remaining endurance for FCR-D Down (until 100%, in minutes)
    def AssetRemainingEnduranceFcrDDown(self, soc_meas):
        return 24 * 60 # not ideal, but have to work for now (since I know there are usages of this method)

    def AssetRemainingSoCEnduranceFcrDDown(self, time_in_minutes):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    def AssetRemainingSoCEnduranceFcrDUp(self, time_in_minutes):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    # This gives remaining time until NEM* is enabled by Fcrd Down
    # * We look for NEM since it will be activated before AEM
    def AssetRemainingSoC2EnergyManagementFcrDDown(self, time_in_minutes):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    # This gives remaining time until NEM* is enabled by Fcrd Up
    # * We look for NEM since it will be activated before AEM
    def AssetRemainingSoC2EnergyManagementFcrDUp(self, time_in_minutes):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    # Remaining endurance for FCR-D Up (until 0%, in minutes)
    def AssetRemainingEnduranceFcrDUp(self, soc_meas):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    # Remaining endurance for FCR-D Down (until NEM, in minutes)
    def MinutesToAssetSoCEnableNEMBoundsFcrDDown(self, soc_meas):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    # Remaining endurance for FCR-D Down (until NEM, in minutes)
    def MinutesToAssetSoCEnableNEMBoundsFcrDUp(self, soc_meas):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    # Remaining endurance for FCR-D Down (until NEM, in minutes)
    def MinutesToAssetSoCEnableAEMBoundsFcrDUp(self, soc_meas):
        raise NotImplementedError("This method is not implemented for non-LER tests")

    def PrintParameters(self):
        pass #raise NotImplementedError("This method is not implemented for non-LER tests")


class FcrDTestSessionStaticNonLER(FcrDTestSessionNonLER):
    def __init__(
            self,
            asset_power: float = None,  # MW
            dbworker = None,
            tempo: float = 1.0,
            verbose: bool = False,
            metadata_log_filename: str = 'metadata_log.csv',
            synchronization_minute = None,
    ):
        super().__init__(
            asset_power=asset_power,
            dbworker=dbworker,
            tempo=tempo,
            verbose=verbose,
            metadata_log_filename=metadata_log_filename,
            synchronization_minute=synchronization_minute
        )

    def InitTests(self, meas_uncertainty=0.001):
        # List of tuples - method, is_fcr_down, min soc, max soc, soc-"cost"(?)

        empty_kwargs = dict()
        self.test_list = [
            (self.fcrdup_ramp_test, False, None, None, empty_kwargs),  # analogous to above
            (self.fcrdup_static_linearity_test, False, None, None, empty_kwargs),
            (self.one_hour_activation_run, None, None, None, empty_kwargs)
        ]

        # Possibly add synchronization
        if isinstance(self.synchronization_minute, int):
            tmp_kwargs = {'synchronization_minute': self.synchronization_minute}
            chrg_tpl = (self.syncronized_start, None, 0.0, 100.0, tmp_kwargs)
            self.test_list.insert(0, chrg_tpl)

        self.run_state = RunState.without_running_state
