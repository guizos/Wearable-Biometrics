from utils.sampling import reservoir_sampling


class ActivitySampler:

    def __init__(self,activity_training,user_training,number_training,labeled_samples):
        self.user_training = user_training
        self.activity_training = activity_training
        self.number_training = number_training
        self.labeled_samples = labeled_samples

    def labeled_training_and_test_samples(self):
        activity_samples = [labeled_sample for labeled_sample in self.labeled_samples if labeled_sample.activity == self.activity_training and labeled_sample.user_id == self.user_training]
        labeled_training = reservoir_sampling(activity_samples, self.number_training)
        labeled_test = [labeled_sample for labeled_sample in self.labeled_samples if labeled_sample not in labeled_training]
        return labeled_training,labeled_test


class RandomSampler:

    def __init__(self,user_id,number_training,labeled_samples):
        self.number_training = number_training
        self.labeled_samples = labeled_samples
        self.user_id = user_id

    def labeled_training_and_test_samples(self):
        user_samples = [labeled_sample for labeled_sample in self.labeled_samples if labeled_sample.user_id == self.user_id]
        labeled_training = reservoir_sampling(user_samples, self.number_training)
        labeled_test = [labeled_sample for labeled_sample in self.labeled_samples if labeled_sample not in labeled_training]
        return labeled_training,labeled_test