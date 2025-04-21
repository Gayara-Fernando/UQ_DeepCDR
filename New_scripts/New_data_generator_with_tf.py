import tensorflow as tf
import numpy as np

# Define an instance for generating batches with Sequence class
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, feature1, feature2, feature3, feature4, feature5, labels, batch_size=32, shuffle=True):
        """
        Initializes the data generator.

        :param feature1, feature2, feature3, feature4, feature5: The 5 different input features.
        :param labels: The target variable.
        :param batch_size: The size of the batch to generate.
        :param shuffle: Whether to shuffle data after each epoch.
        """
        self.feature1 = feature1
        self.feature2 = feature2
        self.feature3 = feature3
        self.feature4 = feature4
        self.feature5 = feature5
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.labels) / self.batch_size))

    def on_epoch_end(self):
        # Shuffling the indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Extract the individual features for the batch
        batch_feature1 = self.feature1[batch_indexes]
        batch_feature2 = self.feature2[batch_indexes]
        batch_feature3 = self.feature3[batch_indexes]
        batch_feature4 = self.feature4[batch_indexes]
        batch_feature5 = self.feature5[batch_indexes]
        
        # Get the corresponding labels for the batch
        batch_labels = self.labels[batch_indexes]

        # Return the individual features and labels
        return ([batch_feature1, batch_feature2, batch_feature3, batch_feature4, batch_feature5], batch_labels)

# Define an instance for generating bootstrap samples (for training data) with Sequence class
class BootstrapGenerator(tf.keras.utils.Sequence):
    def __init__(self, feature1, feature2, feature3, feature4, feature5, labels, batch_size=32, shuffle=True):
        """
        Initializes the data generator with bootstrapping.

        :param feature1, feature2, feature3, feature4, feature5: The 5 different input features.
        :param labels: The target variable.
        :param batch_size: The size of the batch to generate.
        :param shuffle: Whether to shuffle data after each epoch.
        """
        self.feature1 = feature1
        self.feature2 = feature2
        self.feature3 = feature3
        self.feature4 = feature4
        self.feature5 = feature5
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_len = len(self.labels)
        
    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(self.data_len / self.batch_size))

    def on_epoch_end(self):
        # Shuffling is not necessary for bootstrapping, because we sample with replacement
        pass

    def __getitem__(self, index):
        """
        Generate one batch of data using bootstrap sampling.
        """
        # Sample indices with replacement
        batch_indexes = np.random.choice(self.data_len, self.batch_size, replace=True)  # Bootstrapping
        
        # Extract features for the batch
        batch_feature1 = self.feature1[batch_indexes]
        batch_feature2 = self.feature2[batch_indexes]
        batch_feature3 = self.feature3[batch_indexes]
        batch_feature4 = self.feature4[batch_indexes]
        batch_feature5 = self.feature5[batch_indexes]
        
        # Extract the corresponding labels for the batch
        batch_labels = self.labels[batch_indexes]

        # Return the individual features and labels
        return ([batch_feature1, batch_feature2, batch_feature3, batch_feature4, batch_feature5], batch_labels)


# now for prediction 
def batch_predict(model, generator, flatten=True, verbose=False):
    predictions = []
    true_values = []

    for batch_features, batch_labels in generator:
        # Get model predictions for the batch
        batch_predictions = model.predict(batch_features, batch_size=generator.batch_size, verbose=0)
        pred = batch_predictions.flatten()
        y = batch_labels.flatten()
        predictions.extend(pred)
        true_values.extend(y)
        # print("Batch Predict:")
    print(f"Predictions: {len(predictions)}")
    print(f"True: {len(true_values)}")
    return np.array(predictions), np.array(true_values)