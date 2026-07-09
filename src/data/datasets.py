"""
Data loading and preprocessing utilities for all references.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict
import os


class TabularDataset(Dataset):
    """Generic tabular dataset for fairness experiments."""

    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 sensitive: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.sensitive = torch.FloatTensor(sensitive) if sensitive is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"features": self.X[idx], "label": self.y[idx]}
        if self.sensitive is not None:
            item["sensitive"] = self.sensitive[idx]
        return item


def load_adult_income(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load Adult Income dataset.

    Returns:
        X, y, sensitive_attr, feature_names
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame

    # Target
    y = (df['class'] == '>50K').astype(int).values

    # Sensitive attribute: sex
    sensitive_attr = (df['sex'] == 'Male').astype(int).values

    # Features
    feature_cols = [c for c in df.columns if c not in ['class', 'sex']]
    X = pd.get_dummies(df[feature_cols], drop_first=True).values

    return X, y, sensitive_attr, feature_cols


def load_propublica_recidivism(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load ProPublica Recidivism (COMPAS) dataset.

    Returns:
        X, y, sensitive_attr, feature_names
    """
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)

    # Apply the standard COMPAS preprocessing filter used in recidivism studies.
    df = df[(df['days_b_screening_arrest'] <= 30) & 
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != 'O') &
            (df['score_text'] != 'N/A')]

    # Target: two-year recidivism
    y = df['two_year_recid'].values

    # Sensitive attribute: race
    sensitive_attr = (df['race'] == 'Caucasian').astype(int).values

    # Features
    feature_cols = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                    'priors_count', 'c_charge_degree']

    X = pd.get_dummies(df[feature_cols], drop_first=True).values

    return X, y, sensitive_attr, feature_cols


def load_german_credit(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load German Credit dataset.

    Returns:
        X, y, sensitive_attr, feature_names
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame

    # Target: good/bad credit
    y = (df['class'] == 'good').astype(int).values

    # Sensitive attribute: personal_status (male/female)
    sensitive_attr = df['personal_status'].str.contains('male', case=False, na=False).astype(int).values

    # Features
    feature_cols = [c for c in df.columns if c not in ['class', 'personal_status']]
    X = pd.get_dummies(df[feature_cols], drop_first=True).values

    return X, y, sensitive_attr, feature_cols


def load_communities_crime(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load Communities and Crime dataset.

    Returns:
        X, y, sensitive_attr, feature_names
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"

    # Read data (has missing values marked as ?)
    df = pd.read_csv(url, header=None, na_values="?")

    # Drop rows with too many missing values
    df = df.dropna(thresh=len(df.columns) - 20)

    # Fill remaining missing values with median
    df = df.fillna(df.median())

    # Target: violent crime rate (binarized)
    y = (df.iloc[:, -1] > df.iloc[:, -1].median()).astype(int).values

    # Sensitive attribute: race (column 5 is race percentage)
    sensitive_attr = (df.iloc[:, 5] > df.iloc[:, 5].median()).astype(int).values

    # Features (exclude first few identifier columns and target)
    X = df.iloc[:, 5:-1].values

    return X, y, sensitive_attr, list(range(X.shape[1]))


def get_image_loaders(dataset: str, batch_size: int = 128, 
                      data_root: str = "./data") -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test loaders for image datasets.

    Args:
        dataset: "mnist", "fashion_mnist", or "cifar10"
        batch_size: Batch size
        data_root: Root directory for data

    Returns:
        train_loader, test_loader
    """
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    elif dataset == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)

    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def get_tabular_loaders(dataset: str, batch_size: int = 64,
                        test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Get train and test loaders for tabular datasets.

    Args:
        dataset: "adult_income", "propublica_recidivism", "german_credit", "communities_crime"
        batch_size: Batch size
        test_size: Fraction for test set

    Returns:
        train_loader, test_loader, metadata
    """
    loaders = {
        "adult_income": load_adult_income,
        "propublica_recidivism": load_propublica_recidivism,
        "german_credit": load_german_credit,
        "communities_crime": load_communities_crime,
        "compas": load_propublica_recidivism,
    }

    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}")

    X, y, sensitive, feature_names = loaders[dataset]()

    # Split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create datasets
    train_dataset = TabularDataset(X_train, y_train, s_train)
    test_dataset = TabularDataset(X_test, y_test, s_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "feature_names": feature_names,
        "n_features": X_train.shape[1],
        "n_classes": len(np.unique(y)),
        "sensitive_attr_name": dataset.split("_")[0] if "_" in dataset else dataset,
    }

    return train_loader, test_loader, metadata
