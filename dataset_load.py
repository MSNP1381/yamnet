import csv
import librosa
import numpy as np
from keras.utils import Sequence
import os


class AudioSequence(Sequence):

    def __init__(self, prefix: str, classes_list: list, csv_file, batch_size=128, num_classes=10):
        """
        Initialize the audio sequence class.

        Args:
            csv_file: Path to the CSV file containing the audio data.
            batch_size: The batch size to use for training.
            num_classes: The number of classes in the dataset.
        """

        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Read the CSV file and load the audio data.

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            self.data = list(reader)

        # Convert the audio data to NumPy arrays.

        self.data = self.data[1:]
        self.data = [
            (os.path.join(prefix, i), 
             classes_list.index(j if j != 'Sound_Guiatr' else "Sound_Guitar")
             )
              for i, j in self.data]
        self.data = np.array(self.data, dtype=object)
        # self.data = self.data.astype(np.float32)

    def __len__(self):
        """
        Return the number of batches in the dataset.
        """

        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """
        Get a batch of audio data from the dataset.

        Args:
            index: The index of the batch to get.
        """

        # Get the start and end indices of the batch.

        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))

        # Get the audio data for the batch.

        audio_data = self.data[start_idx:end_idx, 0]

        # Load the audio data using Librosa.

        audio_data = [librosa.load(
            file_path, sr=16000, mono=True, dtype='float32') for file_path in audio_data]

        # Convert the audio data to spectrograms.

        # spectrograms = [librosa.feature.melspectrogram(audio, sr=16000, n_mels=128) for audio in audio_data]

        # # Convert the spectrograms to NumPy arrays.

        # spectrograms = np.array(spectrograms, dtype=np.float32)

        # # Normalize the spectrograms.

        # spectrograms = spectrograms / np.max(spectrograms)

        # Get the labels for the batch.

        labels = self.data[start_idx:end_idx, 1]

        # Convert the labels to one-hot vectors.

        labels = np.eye(self.num_classes)[labels]

        return audio_data, labels
