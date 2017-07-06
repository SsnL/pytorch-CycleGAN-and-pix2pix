import numpy as np
import dill as pickle
import os

class MeanEstimator:
    # assumes transformation maps length-n vector to length-n vector
    def __init__(self, transformation = lambda x: x):
        self.transformation = transformation

    def get_estimate(self):
        raise NotImplemented

    # assumes value is an length-n vector
    # if update is set, update estimator with value
    def get_estimated_ratio(self, value, update = True):
        transformed_value = self.transformation(value)
        if update:
            self._record_new_transformed_value(transformed_value)
        estimate = self.get_estimate()
        estimated_ratio = transformed_value / estimate
        return estimate, estimated_ratio

    def _record_new_transformed_value(self, transformed_value):
        raise NotImplemented

    def save(self, save_dir, estimator_label, epoch_label):
        save_filename = '%s_est_%s.pkl' % (epoch_label, estimator_label)
        save_path = os.path.join(save_dir, save_filename)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from(cls, load_dir, estimator_label, epoch_label):
        load_filename = '%s_est_%s.pkl' % (epoch_label, estimator_label)
        load_path = os.path.join(load_dir, load_filename)
        with open(load_path, 'rb') as f:
            return pickle.load(f)

class FullHistoryMeanEstimator(MeanEstimator):
    def __init__(self, transformation = lambda x: x):
        super().__init__(transformation)
        self.total = 0
        self.count = 0

    def get_estimate(self):
        return self.total / self.count

    def _record_new_transformed_value(self, transformed_value):
        self.total += transformed_value.sum()
        self.count += transformed_value.size

class LastKMeanEstimator(MeanEstimator):
    def __init__(self, k, transformation = lambda x: x):
        super().__init__(transformation)
        self.k = k
        self.queue = [None for _ in range(k)]
        self.total = 0
        self.count = 0
        self.at = 0

    def get_estimate(self):
        return self.total / self.count

    def _record_new_transformed_value(self, transformed_value):
        transformed_value_mean = transformed_value.mean()
        if self.count < self.k:
            self.count += 1
        else:
            self.total -= self.queue[self.at]
        self.queue[self.at] = transformed_value_mean
        self.total += transformed_value_mean
        self.at = (self.at + 1) % self.k

class GeometricMeanEstimator(MeanEstimator):
    def __init__(self, decay = 0.9, transformation = lambda x: x):
        super().__init__(transformation)
        assert 0 <= decay < 1, "decay must be in [0, 1)"
        self.decay = decay
        self.estimate = None

    def get_estimate(self):
        return self.estimate

    def _record_new_transformed_value(self, transformed_value):
        if self.estimate is None:
            self.estimate = transformed_value.mean()
        else:
            self.estimate = self.decay * self.estimate + (1 - self.decay) * transformed_value.mean()

def create_estimator(opt):
    if not opt.weight_transform:
        return None
    transform = eval("lambda l:({})".format(opt.weight_transform))
    if opt.weight_mean_estimator == 'last_k':
        return LastKMeanEstimator(int(opt.weight_mean_estimator_arg), transform)
    elif opt.weight_mean_estimator == 'decay':
        return GeometricMeanEstimator(opt.weight_mean_estimator_arg, transform)


