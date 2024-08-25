import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical


# -----------------------------------
#      DATA GENERATOR - 4D CTP
# -----------------------------------
class DataGenerator2D_CTP(Sequence):
    """
        To utilize this data generator:
            1. Images should be already preprocessed, saved as a numpy array of name '{PatientID}_ses-01_ctp_preprocessed_slice_{number}.npz'
            2. Npz files must contain two variables named 'img' and 'label', where 'img' is the input 4D CTP data and 'label' is the corresponding binary lesion segmentation mask (ground truth)
            3. 'img' should be of shape (height, width, n_timepoints)
            4. 'label' should be of shape (height, width, 1)

        Outputs:
            - List of inputs. Elements on the list are 4D CTP and clinical metadata for the specified patient
            - Single element containing the ground truth lesion segmentation mask
        """
    # Initialization
    def __init__(self, list_IDs, imagePath='', dim=(416, 416), batch_size=1, timepoints=32, n_classes=2, features=None, shuffle=True, **kwargs):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.timepoints = timepoints
        self.n_classes = n_classes
        self.features = features
        self.imagePath = imagePath
        self.shuffle = shuffle
        self.on_epoch_end()

    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # Generate indexes of the batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes]  # Find list of IDs
        X, y = self.__data_generation(list_IDs_temp)  # Generate data
        return X, y

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size samples
    def __data_generation(self, list_IDs_temp):
        # Variable initialization
        X = np.empty((self.batch_size, *self.dim, self.timepoints), dtype='float32')
        y = np.empty((self.batch_size, *self.dim, 1), dtype='float32')
        C_continuous = np.empty((self.batch_size, len(self.features[0])), dtype='float32')
        C_categorical = np.empty((self.batch_size, len(self.features[1])), dtype='int32')

        # Generate data according to patient IDs
        for k, ID in enumerate(list_IDs_temp):
            patient = ID.split('_', maxsplit=1)
            patient_data = np.load(self.imagePath + f'{patient[0]}/ctp_preprocessed_slice_{patient[1]}.npz')
            X[k, ] = patient_data['img']  # image shape should be (416, 416, 32)
            y[k, ] = patient_data['label']  # label shape should be (416, 416, 1)

            metadata_pd = pd.read_csv(self.imagePath + f'{patient[0]}/demographic_baseline_imputed.csv')
            metadata_pd['Sex'] = metadata_pd['Sex'].replace({'M': 0, 'F': 1})  # TODO: incorporate this to preprocessing pipeline + min-max normalization for continuous features?
            C_continuous[k, ] = metadata_pd[self.features[0]].to_numpy()  # error will raise if only having one variable. remove to_numpy() in those situations
            C_categorical[k, ] = metadata_pd[self.features[1]].to_numpy()

        # Convert np arrays to lists (to match the model input format)
        X = np.split(X, self.timepoints, axis=3)
        C_continuous = np.split(C_continuous, len(self.features[0]), axis=1)
        C_categorical = np.split(C_categorical, len(self.features[1]), axis=1)
        return [X, C_categorical, C_continuous], to_categorical(y, num_classes=self.n_classes)


# -----------------------------------
#      DATA GENERATOR - 4D CTP
# -----------------------------------
class DataGenerator2D_CTP_Unimodal(Sequence):
    """
        To utilize this data generator:
            1. Images should be already preprocessed, saved as a numpy array of name '{PatientID}_ses-01_ctp_preprocessed_slice_{number}.npz'
            2. Npz files must contain two variables named 'img' and 'label', where 'img' is the input 4D CTP data and 'label' is the corresponding binary lesion segmentation mask (ground truth)
            3. 'img' should be of shape (height, width, n_timepoints)
            4. 'label' should be of shape (height, width, 1)

        Outputs:
            - List of inputs. Elements on the list are 4D CTP and clinical metadata for the specified patient
            - Single element containing the ground truth lesion segmentation mask
        """
    # Initialization
    def __init__(self, list_IDs, imagePath='', dim=(416, 416), batch_size=1, timepoints=32, n_classes=2, features=None, shuffle=True, **kwargs):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.timepoints = timepoints
        self.n_classes = n_classes
        self.features = features
        self.imagePath = imagePath
        self.shuffle = shuffle
        self.on_epoch_end()

    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # Generate indexes of the batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes]  # Find list of IDs
        X, y = self.__data_generation(list_IDs_temp)  # Generate data
        return X, y

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size samples
    def __data_generation(self, list_IDs_temp):
        # Variable initialization
        X = np.empty((self.batch_size, *self.dim, self.timepoints), dtype='float32')
        y = np.empty((self.batch_size, *self.dim, 1), dtype='float32')

        # Generate data according to patient IDs
        for k, ID in enumerate(list_IDs_temp):
            patient = ID.split('_', maxsplit=1)
            patient_data = np.load(self.imagePath + f'{patient[0]}/ctp_preprocessed_slice_{patient[1]}.npz')
            X[k, ] = patient_data['img']  # image shape should be (416, 416, 32)
            y[k, ] = patient_data['label']  # label shape should be (416, 416, 1)

        # Convert np arrays to lists (to match the model input format)
        X = np.split(X, self.timepoints, axis=3)
        return X, to_categorical(y, num_classes=self.n_classes)
