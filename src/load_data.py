"""Utilities for loading the UCI SECOM dataset."""

from ucimlrepo import fetch_ucirepo


SECOM_DATASET_ID = 179


def load_secom_dataset():
    """Fetch and return the complete UCI SECOM dataset object."""
    return fetch_ucirepo(id=SECOM_DATASET_ID)


def split_secom_data(dataset):
    """Return SECOM features and target without changing the downloaded values."""
    if dataset.data.features is not None and dataset.data.targets is not None:
        return dataset.data.features, dataset.data.targets

    # Dataset 179 currently provides the complete table but no variable schema
    # through ucimlrepo. The official SECOM description identifies `class` as
    # the Pass/Fail target and the remaining 591 columns as input features.
    original = dataset.data.original
    if original is None or "class" not in original.columns:
        raise ValueError("UCI SECOM data did not include the expected `class` column.")
    return original.drop(columns="class"), original[["class"]]


def load_secom_data():
    """Fetch the SECOM feature matrix and target values as pandas objects."""
    return split_secom_data(load_secom_dataset())


def load_secom_metadata():
    """Fetch the SECOM metadata and variable descriptions."""
    dataset = load_secom_dataset()
    return dataset.metadata, dataset.variables
