"""Advanced tabular models such as Multi-View Gated MLP or FT-Transformer."""

from dataclasses import dataclass
import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import CATEGORICAL_COLUMNS, FEATURE_GROUPS, LABELS, RANDOM_STATE, TARGET_COL


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dense_array(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _build_view_preprocessor(columns):
    categorical_features = [c for c in columns if c in CATEGORICAL_COLUMNS]
    numeric_features = [c for c in columns if c not in categorical_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


@dataclass
class MultiViewDataBundle:
    train_views: dict
    val_views: dict
    test_views: dict
    train_labels_idx: np.ndarray
    val_labels_idx: np.ndarray
    test_labels_idx: np.ndarray
    train_labels_text: np.ndarray
    val_labels_text: np.ndarray
    test_labels_text: np.ndarray
    view_dims: dict


@dataclass
class FTDataBundle:
    train_cat: np.ndarray
    val_cat: np.ndarray
    test_cat: np.ndarray
    train_num: np.ndarray
    val_num: np.ndarray
    test_num: np.ndarray
    train_labels_idx: np.ndarray
    val_labels_idx: np.ndarray
    test_labels_idx: np.ndarray
    train_labels_text: np.ndarray
    val_labels_text: np.ndarray
    test_labels_text: np.ndarray
    cat_cardinalities: list
    num_feature_names: list
    cat_feature_names: list
    cat_view_ids: np.ndarray
    num_view_ids: np.ndarray
    view_names: list
    numeric_bin_edges: np.ndarray | None


def prepare_multiview_data(train_df, test_df, val_size=0.2, random_state=RANDOM_STATE, feature_groups=None):
    feature_groups = feature_groups or FEATURE_GROUPS
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df[TARGET_COL],
        random_state=random_state,
    )

    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    train_labels_text = train_split[TARGET_COL].astype(str).to_numpy()
    val_labels_text = val_split[TARGET_COL].astype(str).to_numpy()
    test_labels_text = test_df[TARGET_COL].astype(str).to_numpy()

    train_views = {}
    val_views = {}
    test_views = {}
    view_dims = {}

    for view_name, columns in feature_groups.items():
        preprocessor = _build_view_preprocessor(columns)
        train_matrix = _dense_array(preprocessor.fit_transform(train_split[columns]))
        val_matrix = _dense_array(preprocessor.transform(val_split[columns]))
        test_matrix = _dense_array(preprocessor.transform(test_df[columns]))

        train_views[view_name] = train_matrix
        val_views[view_name] = val_matrix
        test_views[view_name] = test_matrix
        view_dims[view_name] = train_matrix.shape[1]

    return MultiViewDataBundle(
        train_views=train_views,
        val_views=val_views,
        test_views=test_views,
        train_labels_idx=np.array([label_to_idx[v] for v in train_labels_text], dtype=np.int64),
        val_labels_idx=np.array([label_to_idx[v] for v in val_labels_text], dtype=np.int64),
        test_labels_idx=np.array([label_to_idx[v] for v in test_labels_text], dtype=np.int64),
        train_labels_text=train_labels_text,
        val_labels_text=val_labels_text,
        test_labels_text=test_labels_text,
        view_dims=view_dims,
    )


class MultiViewDataset(Dataset):
    def __init__(self, view_arrays, labels):
        self.view_names = list(view_arrays.keys())
        self.view_tensors = [torch.tensor(view_arrays[name], dtype=torch.float32) for name in self.view_names]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        view_values = [tensor[idx] for tensor in self.view_tensors]
        return (*view_values, self.labels[idx])


class FTTabularDataset(Dataset):
    def __init__(self, cat_array, num_array, labels):
        self.cat_tensor = torch.tensor(cat_array, dtype=torch.long)
        self.num_tensor = torch.tensor(num_array, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat_tensor[idx], self.num_tensor[idx], self.labels[idx]


class MultiViewGatedMLP(nn.Module):
    def __init__(self, view_dims, hidden_dim=64, dropout=0.2, num_classes=3):
        super().__init__()
        self.view_names = list(view_dims.keys())
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for name, dim in view_dims.items()
            }
        )
        num_views = len(self.view_names)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * num_views, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_views),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (num_views + 1) + num_views, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, *view_inputs):
        encoded = [self.encoders[name](tensor) for name, tensor in zip(self.view_names, view_inputs)]
        stacked = torch.stack(encoded, dim=1)
        flat = stacked.reshape(stacked.size(0), -1)
        gate_logits = self.gate_net(flat)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        classifier_input = torch.cat([flat, fused, gate_weights], dim=1)
        logits = self.classifier(classifier_input)
        return logits, gate_weights


class TabMStyleMultiExpertMLP(nn.Module):
    def __init__(self, view_dims, hidden_dim=64, dropout=0.1, num_experts=8, expert_hidden_dim=96, num_classes=3):
        super().__init__()
        self.view_names = list(view_dims.keys())
        self.num_experts = num_experts
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for name, dim in view_dims.items()
            }
        )
        num_views = len(self.view_names)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * num_views, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_views),
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_dim * (num_views + 1) + num_views, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.expert_scale = nn.Parameter(torch.zeros(num_experts, expert_hidden_dim))
        self.expert_bias = nn.Parameter(torch.zeros(num_experts, expert_hidden_dim))
        self.expert_head = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.expert_out = nn.Linear(expert_hidden_dim, num_classes)

    def forward(self, *view_inputs):
        encoded = [self.encoders[name](tensor) for name, tensor in zip(self.view_names, view_inputs)]
        stacked = torch.stack(encoded, dim=1)
        flat = stacked.reshape(stacked.size(0), -1)
        gate_logits = self.gate_net(flat)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        classifier_input = torch.cat([flat, fused, gate_weights], dim=1)
        shared = self.shared_projection(classifier_input)

        expert_input = shared.unsqueeze(1) * (1 + self.expert_scale.unsqueeze(0)) + self.expert_bias.unsqueeze(0)
        expert_hidden = self.expert_head(expert_input)
        expert_logits = self.expert_out(expert_hidden)
        return expert_logits, gate_weights


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_num_features, d_token))

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class PeriodicNumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, d_token, n_frequencies=8):
        super().__init__()
        self.n_num_features = n_num_features
        self.n_frequencies = n_frequencies
        self.frequencies = nn.Parameter(torch.randn(n_num_features, n_frequencies) * 0.5)
        self.bias = nn.Parameter(torch.zeros(n_num_features, n_frequencies))
        self.projection = nn.Linear(1 + 2 * n_frequencies, d_token)

    def forward(self, x_num):
        if x_num.shape[1] == 0:
            return x_num.new_zeros((x_num.shape[0], 0, self.projection.out_features))
        expanded = x_num.unsqueeze(-1)
        angles = expanded * self.frequencies.unsqueeze(0) + self.bias.unsqueeze(0)
        features = torch.cat([expanded, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.projection(features)


class PiecewiseNumericalFeatureTokenizer(nn.Module):
    def __init__(self, numeric_bin_edges, d_token):
        super().__init__()
        numeric_bin_edges = torch.tensor(numeric_bin_edges, dtype=torch.float32)
        self.register_buffer("numeric_bin_edges", numeric_bin_edges, persistent=False)
        self.n_num_features = numeric_bin_edges.shape[0]
        self.n_bins = numeric_bin_edges.shape[1] - 1
        basis_dim = self.n_bins + 1
        self.weight = nn.Parameter(torch.randn(self.n_num_features, basis_dim, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.n_num_features, d_token))

    def forward(self, x_num):
        if x_num.shape[1] == 0:
            return x_num.new_zeros((x_num.shape[0], 0, self.weight.shape[-1]))

        batch_size, n_features = x_num.shape
        basis_dim = self.n_bins + 1
        basis = x_num.new_zeros((batch_size, n_features, basis_dim))
        eps = 1e-6

        for feature_idx in range(n_features):
            edges = self.numeric_bin_edges[feature_idx]
            bucket_idx = torch.bucketize(x_num[:, feature_idx], edges[1:-1])
            left = edges[bucket_idx]
            right = edges[bucket_idx + 1]
            t = ((x_num[:, feature_idx] - left) / (right - left + eps)).clamp(0.0, 1.0)

            basis[:, feature_idx, :].scatter_(1, bucket_idx.unsqueeze(1), (1.0 - t).unsqueeze(1))
            basis[:, feature_idx, :].scatter_add_(
                1,
                (bucket_idx + 1).unsqueeze(1),
                t.unsqueeze(1),
            )

        return torch.einsum("bfp,fpd->bfd", basis, self.weight) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_num_features,
        d_token=32,
        n_blocks=2,
        attention_n_heads=4,
        attention_dropout=0.2,
        ff_dropout=0.2,
        num_classes=3,
    ):
        super().__init__()
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        self.num_tokenizer = NumericalFeatureTokenizer(n_num_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_n_heads,
            dim_feedforward=d_token * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_token * 2, num_classes),
        )

    def forward(self, x_cat, x_num):
        tokens = []

        if len(self.cat_embeddings) > 0:
            cat_tokens = [embedding(x_cat[:, idx]) for idx, embedding in enumerate(self.cat_embeddings)]
            tokens.append(torch.stack(cat_tokens, dim=1))

        if x_num.shape[1] > 0:
            num_tokens = self.num_tokenizer(x_num)
            tokens.append(num_tokens)

        if not tokens:
            raise ValueError("FTTransformer requires at least one categorical or numerical feature.")

        token_sequence = torch.cat(tokens, dim=1)
        cls_token = self.cls_token.expand(token_sequence.size(0), -1, -1)
        x = torch.cat([cls_token, token_sequence], dim=1)
        x = self.transformer(x)
        cls_representation = x[:, 0]
        logits = self.head(cls_representation)
        return logits


class PiecewiseFTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        numeric_bin_edges,
        d_token=48,
        n_blocks=2,
        attention_n_heads=4,
        attention_dropout=0.2,
        ff_dropout=0.2,
        num_classes=3,
    ):
        super().__init__()
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        self.num_tokenizer = PiecewiseNumericalFeatureTokenizer(numeric_bin_edges=numeric_bin_edges, d_token=d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_n_heads,
            dim_feedforward=d_token * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_token * 2, num_classes),
        )

    def forward(self, x_cat, x_num):
        tokens = []

        if len(self.cat_embeddings) > 0:
            cat_tokens = [embedding(x_cat[:, idx]) for idx, embedding in enumerate(self.cat_embeddings)]
            tokens.append(torch.stack(cat_tokens, dim=1))

        if x_num.shape[1] > 0:
            tokens.append(self.num_tokenizer(x_num))

        if not tokens:
            raise ValueError("PiecewiseFTTransformer requires at least one categorical or numerical feature.")

        token_sequence = torch.cat(tokens, dim=1)
        cls_token = self.cls_token.expand(token_sequence.size(0), -1, -1)
        x = torch.cat([cls_token, token_sequence], dim=1)
        x = self.transformer(x)
        cls_representation = x[:, 0]
        logits = self.head(cls_representation)
        return logits


class ViewAwareFTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_num_features,
        cat_view_ids,
        num_view_ids,
        n_views,
        d_token=48,
        n_blocks=2,
        attention_n_heads=4,
        attention_dropout=0.2,
        ff_dropout=0.2,
        n_num_frequencies=8,
        num_classes=3,
    ):
        super().__init__()
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        self.num_tokenizer = PeriodicNumericalFeatureTokenizer(
            n_num_features=n_num_features,
            d_token=d_token,
            n_frequencies=n_num_frequencies,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        self.column_embedding = nn.Embedding(1 + len(cat_cardinalities) + n_num_features, d_token)
        self.view_embedding = nn.Embedding(n_views + 1, d_token)
        self.type_embedding = nn.Embedding(3, d_token)

        total_cat = len(cat_cardinalities)
        self.register_buffer("cat_column_ids", torch.arange(1, 1 + total_cat, dtype=torch.long), persistent=False)
        self.register_buffer(
            "num_column_ids",
            torch.arange(1 + total_cat, 1 + total_cat + n_num_features, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("cat_view_ids", torch.tensor(cat_view_ids, dtype=torch.long), persistent=False)
        self.register_buffer("num_view_ids", torch.tensor(num_view_ids, dtype=torch.long), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_n_heads,
            dim_feedforward=d_token * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_token * 2, num_classes),
        )

    def forward(self, x_cat, x_num):
        tokens = []

        if len(self.cat_embeddings) > 0:
            cat_tokens = [embedding(x_cat[:, idx]) for idx, embedding in enumerate(self.cat_embeddings)]
            cat_tokens = torch.stack(cat_tokens, dim=1)
            cat_tokens = (
                cat_tokens
                + self.column_embedding(self.cat_column_ids).unsqueeze(0)
                + self.view_embedding(self.cat_view_ids).unsqueeze(0)
                + self.type_embedding(torch.zeros_like(self.cat_view_ids)).unsqueeze(0)
            )
            tokens.append(cat_tokens)

        if x_num.shape[1] > 0:
            num_tokens = self.num_tokenizer(x_num)
            num_tokens = (
                num_tokens
                + self.column_embedding(self.num_column_ids).unsqueeze(0)
                + self.view_embedding(self.num_view_ids).unsqueeze(0)
                + self.type_embedding(torch.ones_like(self.num_view_ids)).unsqueeze(0)
            )
            tokens.append(num_tokens)

        if not tokens:
            raise ValueError("ViewAwareFTTransformer requires at least one categorical or numerical feature.")

        token_sequence = torch.cat(tokens, dim=1)
        cls_token = self.cls_token.expand(token_sequence.size(0), -1, -1)
        cls_token = cls_token + self.type_embedding(
            torch.full((1,), 2, device=token_sequence.device, dtype=torch.long)
        ).unsqueeze(0)
        x = torch.cat([cls_token, token_sequence], dim=1)
        x = self.transformer(x)
        cls_representation = x[:, 0]
        logits = self.head(cls_representation)
        return logits


class ViewWiseGatedFTTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_num_features,
        cat_view_ids,
        num_view_ids,
        n_views,
        d_token=48,
        n_blocks=2,
        attention_n_heads=4,
        attention_dropout=0.2,
        ff_dropout=0.2,
        n_num_frequencies=8,
        num_classes=3,
    ):
        super().__init__()
        self.n_views = n_views
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        self.num_tokenizer = PeriodicNumericalFeatureTokenizer(
            n_num_features=n_num_features,
            d_token=d_token,
            n_frequencies=n_num_frequencies,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        self.column_embedding = nn.Embedding(1 + len(cat_cardinalities) + n_num_features, d_token)
        self.view_embedding = nn.Embedding(n_views + 1, d_token)
        self.type_embedding = nn.Embedding(3, d_token)

        total_cat = len(cat_cardinalities)
        self.register_buffer("cat_column_ids", torch.arange(1, 1 + total_cat, dtype=torch.long), persistent=False)
        self.register_buffer(
            "num_column_ids",
            torch.arange(1 + total_cat, 1 + total_cat + n_num_features, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("cat_view_ids", torch.tensor(cat_view_ids, dtype=torch.long), persistent=False)
        self.register_buffer("num_view_ids", torch.tensor(num_view_ids, dtype=torch.long), persistent=False)
        combined_view_ids = np.concatenate([cat_view_ids, num_view_ids], axis=0) if n_num_features or total_cat else []
        self.register_buffer(
            "token_view_ids",
            torch.tensor(combined_view_ids, dtype=torch.long),
            persistent=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_n_heads,
            dim_feedforward=d_token * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.view_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_token),
                    nn.Linear(d_token, d_token),
                    nn.ReLU(),
                    nn.Dropout(ff_dropout),
                )
                for _ in range(n_views)
            ]
        )
        self.gate_net = nn.Sequential(
            nn.Linear(d_token * n_views + d_token, d_token),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_token, n_views),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token * (n_views + 2) + n_views),
            nn.Linear(d_token * (n_views + 2) + n_views, d_token * 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_token * 2, num_classes),
        )

    def _build_tokens(self, x_cat, x_num):
        tokens = []

        if len(self.cat_embeddings) > 0:
            cat_tokens = [embedding(x_cat[:, idx]) for idx, embedding in enumerate(self.cat_embeddings)]
            cat_tokens = torch.stack(cat_tokens, dim=1)
            cat_tokens = (
                cat_tokens
                + self.column_embedding(self.cat_column_ids).unsqueeze(0)
                + self.view_embedding(self.cat_view_ids).unsqueeze(0)
                + self.type_embedding(torch.zeros_like(self.cat_view_ids)).unsqueeze(0)
            )
            tokens.append(cat_tokens)

        if x_num.shape[1] > 0:
            num_tokens = self.num_tokenizer(x_num)
            num_tokens = (
                num_tokens
                + self.column_embedding(self.num_column_ids).unsqueeze(0)
                + self.view_embedding(self.num_view_ids).unsqueeze(0)
                + self.type_embedding(torch.ones_like(self.num_view_ids)).unsqueeze(0)
            )
            tokens.append(num_tokens)

        if not tokens:
            raise ValueError("ViewWiseGatedFTTransformer requires at least one categorical or numerical feature.")

        return torch.cat(tokens, dim=1)

    def _pool_views(self, encoded_tokens):
        pooled_views = []
        for view_idx in range(1, self.n_views + 1):
            mask = self.token_view_ids == view_idx
            if mask.sum() == 0:
                pooled = encoded_tokens.new_zeros((encoded_tokens.size(0), encoded_tokens.size(-1)))
            else:
                pooled = encoded_tokens[:, mask, :].mean(dim=1)
            pooled_views.append(self.view_projectors[view_idx - 1](pooled))
        return torch.stack(pooled_views, dim=1)

    def forward(self, x_cat, x_num):
        token_sequence = self._build_tokens(x_cat, x_num)
        cls_token = self.cls_token.expand(token_sequence.size(0), -1, -1)
        cls_token = cls_token + self.type_embedding(
            torch.full((1,), 2, device=token_sequence.device, dtype=torch.long)
        ).unsqueeze(0)
        x = torch.cat([cls_token, token_sequence], dim=1)
        x = self.transformer(x)

        cls_representation = x[:, 0]
        encoded_tokens = x[:, 1:, :]
        pooled_views = self._pool_views(encoded_tokens)
        flat_views = pooled_views.reshape(pooled_views.size(0), -1)
        gate_input = torch.cat([cls_representation, flat_views], dim=1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused = (pooled_views * gate_weights.unsqueeze(-1)).sum(dim=1)
        classifier_input = torch.cat([cls_representation, flat_views, fused, gate_weights], dim=1)
        logits = self.head(classifier_input)
        return logits, gate_weights


class ViewWiseAttentionGatedFTTransformer(ViewWiseGatedFTTransformer):
    def __init__(
        self,
        cat_cardinalities,
        n_num_features,
        cat_view_ids,
        num_view_ids,
        n_views,
        d_token=48,
        n_blocks=2,
        attention_n_heads=4,
        attention_dropout=0.2,
        ff_dropout=0.2,
        n_num_frequencies=8,
        num_classes=3,
    ):
        super().__init__(
            cat_cardinalities=cat_cardinalities,
            n_num_features=n_num_features,
            cat_view_ids=cat_view_ids,
            num_view_ids=num_view_ids,
            n_views=n_views,
            d_token=d_token,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ff_dropout=ff_dropout,
            n_num_frequencies=n_num_frequencies,
            num_classes=num_classes,
        )
        self.view_attention_poolers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_token),
                    nn.Linear(d_token, d_token),
                    nn.Tanh(),
                    nn.Linear(d_token, 1),
                )
                for _ in range(n_views)
            ]
        )

    def _pool_views(self, encoded_tokens):
        pooled_views = []
        for view_idx in range(1, self.n_views + 1):
            mask = self.token_view_ids == view_idx
            if mask.sum() == 0:
                pooled = encoded_tokens.new_zeros((encoded_tokens.size(0), encoded_tokens.size(-1)))
            else:
                view_tokens = encoded_tokens[:, mask, :]
                attention_scores = self.view_attention_poolers[view_idx - 1](view_tokens).squeeze(-1)
                attention_weights = torch.softmax(attention_scores, dim=1)
                pooled = (view_tokens * attention_weights.unsqueeze(-1)).sum(dim=1)
            pooled_views.append(self.view_projectors[view_idx - 1](pooled))
        return torch.stack(pooled_views, dim=1)


class TromptStylePromptTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        n_num_features,
        d_token=48,
        n_prompts=8,
        n_blocks=2,
        attention_n_heads=4,
        attention_dropout=0.2,
        ff_dropout=0.2,
        n_num_frequencies=8,
        num_classes=3,
    ):
        super().__init__()
        self.n_prompts = n_prompts
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        self.num_tokenizer = PeriodicNumericalFeatureTokenizer(
            n_num_features=n_num_features,
            d_token=d_token,
            n_frequencies=n_num_frequencies,
        )
        total_features = len(cat_cardinalities) + n_num_features
        self.column_embedding = nn.Embedding(total_features, d_token)
        self.type_embedding = nn.Embedding(2, d_token)

        self.prompt_tokens = nn.Parameter(torch.randn(n_prompts, d_token) * 0.02)
        self.prompt_query = nn.Linear(d_token, d_token)
        self.feature_key = nn.Linear(d_token, d_token)
        self.feature_value = nn.Linear(d_token, d_token)
        self.prompt_update = nn.Sequential(
            nn.Linear(d_token * 2, d_token),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_n_heads,
            dim_feedforward=d_token * 4,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_token * 2, num_classes),
        )

        self.register_buffer(
            "column_ids",
            torch.arange(total_features, dtype=torch.long),
            persistent=False,
        )
        if len(cat_cardinalities) > 0:
            self.register_buffer(
                "cat_type_ids",
                torch.zeros(len(cat_cardinalities), dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer("cat_type_ids", torch.zeros(0, dtype=torch.long), persistent=False)
        if n_num_features > 0:
            self.register_buffer(
                "num_type_ids",
                torch.ones(n_num_features, dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer("num_type_ids", torch.ones(0, dtype=torch.long), persistent=False)

    def _build_feature_tokens(self, x_cat, x_num):
        tokens = []
        offset = 0

        if len(self.cat_embeddings) > 0:
            cat_tokens = [embedding(x_cat[:, idx]) for idx, embedding in enumerate(self.cat_embeddings)]
            cat_tokens = torch.stack(cat_tokens, dim=1)
            cat_col_ids = self.column_ids[offset: offset + len(self.cat_embeddings)]
            cat_tokens = cat_tokens + self.column_embedding(cat_col_ids).unsqueeze(0)
            cat_tokens = cat_tokens + self.type_embedding(self.cat_type_ids).unsqueeze(0)
            tokens.append(cat_tokens)
            offset += len(self.cat_embeddings)

        if x_num.shape[1] > 0:
            num_tokens = self.num_tokenizer(x_num)
            num_col_ids = self.column_ids[offset: offset + x_num.shape[1]]
            num_tokens = num_tokens + self.column_embedding(num_col_ids).unsqueeze(0)
            num_tokens = num_tokens + self.type_embedding(self.num_type_ids).unsqueeze(0)
            tokens.append(num_tokens)

        if not tokens:
            raise ValueError("TromptStylePromptTransformer requires at least one categorical or numerical feature.")
        return torch.cat(tokens, dim=1)

    def forward(self, x_cat, x_num):
        feature_tokens = self._build_feature_tokens(x_cat, x_num)
        prompts = self.prompt_tokens.unsqueeze(0).expand(feature_tokens.size(0), -1, -1)

        queries = self.prompt_query(prompts)
        keys = self.feature_key(feature_tokens)
        values = self.feature_value(feature_tokens)
        scores = torch.matmul(queries, keys.transpose(1, 2)) / np.sqrt(feature_tokens.size(-1))
        importance = torch.softmax(scores, dim=-1)
        prompt_summaries = torch.matmul(importance, values)
        updated_prompts = self.prompt_update(torch.cat([prompts, prompt_summaries], dim=-1))
        encoded_prompts = self.transformer(updated_prompts)
        pooled = encoded_prompts.mean(dim=1)
        logits = self.head(pooled)
        return logits, importance.mean(dim=1)


def create_dataloaders(data_bundle, batch_size=128):
    train_dataset = MultiViewDataset(data_bundle.train_views, data_bundle.train_labels_idx)
    val_dataset = MultiViewDataset(data_bundle.val_views, data_bundle.val_labels_idx)
    test_dataset = MultiViewDataset(data_bundle.test_views, data_bundle.test_labels_idx)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def create_ft_dataloaders(data_bundle, batch_size=128):
    train_dataset = FTTabularDataset(data_bundle.train_cat, data_bundle.train_num, data_bundle.train_labels_idx)
    val_dataset = FTTabularDataset(data_bundle.val_cat, data_bundle.val_num, data_bundle.val_labels_idx)
    test_dataset = FTTabularDataset(data_bundle.test_cat, data_bundle.test_num, data_bundle.test_labels_idx)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def _feature_view_ids(feature_names):
    view_names = list(FEATURE_GROUPS.keys())
    feature_to_view = {}
    for idx, view_name in enumerate(view_names, start=1):
        for col in FEATURE_GROUPS[view_name]:
            feature_to_view[col] = idx
    view_ids = np.array([feature_to_view.get(col, 0) for col in feature_names], dtype=np.int64)
    return view_ids, view_names


def _compute_numeric_bin_edges(train_df, num_feature_names, n_bins):
    if not num_feature_names:
        return None

    all_edges = []
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    for col in num_feature_names:
        values = train_df[col].to_numpy(dtype=np.float32)
        values = values[~np.isnan(values)]
        if values.size == 0:
            values = np.array([0.0], dtype=np.float32)
        q = np.quantile(values, quantiles).astype(np.float32)
        if np.unique(q).size < 2:
            center = float(values.mean())
            q = np.linspace(center - 1e-3, center + 1e-3, n_bins + 1, dtype=np.float32)
        elif np.any(np.diff(q) <= 0):
            q = np.linspace(float(values.min()), float(values.max()), n_bins + 1, dtype=np.float32)
            if np.allclose(q[0], q[-1]):
                q = np.linspace(float(values.min()) - 1e-3, float(values.max()) + 1e-3, n_bins + 1, dtype=np.float32)
        all_edges.append(q)
    return np.stack(all_edges, axis=0)


def _run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        *views, labels = batch
        views = [tensor.to(device) for tensor in views]
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(*views)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def _run_epoch_ft(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_cat, x_num, labels in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(x_cat, x_num)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def predict_multiview(model, loader, device):
    model.eval()
    probs = []
    preds = []
    gates = []

    with torch.no_grad():
        for batch in loader:
            *views, labels = batch
            views = [tensor.to(device) for tensor in views]
            logits, gate_weights = model(*views)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            probs.append(prob.cpu().numpy())
            preds.append(pred.cpu().numpy())
            gates.append(gate_weights.cpu().numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(probs, axis=0),
        np.concatenate(gates, axis=0),
    )


def predict_tabm_style(model, loader, device):
    model.eval()
    probs = []
    preds = []
    gates = []

    with torch.no_grad():
        for batch in loader:
            *views, _ = batch
            views = [tensor.to(device) for tensor in views]
            expert_logits, gate_weights = model(*views)
            expert_probs = torch.softmax(expert_logits, dim=-1)
            mean_prob = expert_probs.mean(dim=1)
            pred = mean_prob.argmax(dim=1)

            probs.append(mean_prob.cpu().numpy())
            preds.append(pred.cpu().numpy())
            gates.append(gate_weights.cpu().numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(probs, axis=0),
        np.concatenate(gates, axis=0),
    )


def predict_ft_transformer(model, loader, device):
    model.eval()
    probs = []
    preds = []

    with torch.no_grad():
        for x_cat, x_num, _ in loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            logits = model(x_cat, x_num)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            probs.append(prob.cpu().numpy())
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(probs, axis=0)


def predict_gated_ft_transformer(model, loader, device):
    model.eval()
    probs = []
    preds = []
    gates = []

    with torch.no_grad():
        for x_cat, x_num, _ in loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            logits, gate_weights = model(x_cat, x_num)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            probs.append(prob.cpu().numpy())
            preds.append(pred.cpu().numpy())
            gates.append(gate_weights.cpu().numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(probs, axis=0),
        np.concatenate(gates, axis=0),
    )


def prepare_ft_transformer_data(train_df, test_df, val_size=0.2, random_state=RANDOM_STATE, n_bins=8):
    feature_df = train_df.drop(columns=[TARGET_COL])
    all_columns = feature_df.columns.tolist()
    cat_feature_names = [col for col in all_columns if col in CATEGORICAL_COLUMNS]
    num_feature_names = [col for col in all_columns if col not in CATEGORICAL_COLUMNS]

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df[TARGET_COL],
        random_state=random_state,
    )

    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    train_labels_text = train_split[TARGET_COL].astype(str).to_numpy()
    val_labels_text = val_split[TARGET_COL].astype(str).to_numpy()
    test_labels_text = test_df[TARGET_COL].astype(str).to_numpy()

    cat_cardinalities = []
    category_maps = {}
    for col in cat_feature_names:
        categories = train_split[col].astype(str).fillna("__missing__").unique().tolist()
        category_maps[col] = {value: idx + 1 for idx, value in enumerate(categories)}
        cat_cardinalities.append(len(categories) + 1)

    def encode_categorical(df):
        if not cat_feature_names:
            return np.zeros((len(df), 0), dtype=np.int64)
        encoded = []
        for col in cat_feature_names:
            mapping = category_maps[col]
            values = df[col].astype(str).fillna("__missing__").map(mapping).fillna(0).astype(int).to_numpy()
            encoded.append(values)
        return np.stack(encoded, axis=1).astype(np.int64)

    train_num_df = train_split[num_feature_names].copy()
    val_num_df = val_split[num_feature_names].copy()
    test_num_df = test_df[num_feature_names].copy()

    if num_feature_names:
        medians = train_num_df.median()
        means = train_num_df.fillna(medians).mean()
        stds = train_num_df.fillna(medians).std().replace(0, 1.0)

        def encode_numeric(df):
            normalized = df[num_feature_names].fillna(medians)
            normalized = (normalized - means) / stds
            return normalized.to_numpy(dtype=np.float32)

        train_num = encode_numeric(train_split)
        val_num = encode_numeric(val_split)
        test_num = encode_numeric(test_df)
    else:
        train_num = np.zeros((len(train_split), 0), dtype=np.float32)
        val_num = np.zeros((len(val_split), 0), dtype=np.float32)
        test_num = np.zeros((len(test_df), 0), dtype=np.float32)

    cat_view_ids, view_names = _feature_view_ids(cat_feature_names)
    num_view_ids, _ = _feature_view_ids(num_feature_names)
    numeric_bin_edges = _compute_numeric_bin_edges(train_split, num_feature_names, n_bins=n_bins)

    return FTDataBundle(
        train_cat=encode_categorical(train_split),
        val_cat=encode_categorical(val_split),
        test_cat=encode_categorical(test_df),
        train_num=train_num,
        val_num=val_num,
        test_num=test_num,
        train_labels_idx=np.array([label_to_idx[v] for v in train_labels_text], dtype=np.int64),
        val_labels_idx=np.array([label_to_idx[v] for v in val_labels_text], dtype=np.int64),
        test_labels_idx=np.array([label_to_idx[v] for v in test_labels_text], dtype=np.int64),
        train_labels_text=train_labels_text,
        val_labels_text=val_labels_text,
        test_labels_text=test_labels_text,
        cat_cardinalities=cat_cardinalities,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
        cat_view_ids=cat_view_ids,
        num_view_ids=num_view_ids,
        view_names=view_names,
        numeric_bin_edges=numeric_bin_edges,
    )


def train_multiview_gated_mlp(
    train_df,
    test_df,
    feature_groups=None,
    seed=RANDOM_STATE,
    hidden_dim=64,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=80,
    patience=12,
    class_weight_multipliers=None,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_groups = feature_groups or FEATURE_GROUPS
    data_bundle = prepare_multiview_data(train_df, test_df, feature_groups=feature_groups, random_state=seed)
    dataloaders = create_dataloaders(data_bundle, batch_size=batch_size)

    model = MultiViewGatedMLP(
        view_dims=data_bundle.view_dims,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    if class_weight_multipliers:
        multipliers = np.array([class_weight_multipliers.get(label, 1.0) for label in LABELS], dtype=np.float32)
        class_weights = class_weights * multipliers
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, dataloaders["train"], criterion, optimizer, device)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloaders["val"]:
                *views, labels = batch
                views = [tensor.to(device) for tensor in views]
                labels = labels.to(device)
                logits, _ = model(*views)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_idx, test_prob, test_gates = predict_multiview(model, dataloaders["test"], device)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])

    gate_frame = pd.DataFrame(test_gates, columns=list(data_bundle.view_dims.keys()))
    gate_frame["true_label"] = data_bundle.test_labels_text

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "class_weights": {label: float(weight) for label, weight in zip(LABELS, class_weights)},
        "class_weight_multipliers": class_weight_multipliers or {},
        "view_dims": data_bundle.view_dims,
        "history": history,
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
        "gate_frame": gate_frame,
    }


def train_tabm_style_multiexpert_mlp(
    train_df,
    test_df,
    feature_groups=None,
    seed=RANDOM_STATE,
    hidden_dim=64,
    dropout=0.1,
    num_experts=8,
    expert_hidden_dim=96,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=80,
    patience=12,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_groups = feature_groups or FEATURE_GROUPS
    data_bundle = prepare_multiview_data(train_df, test_df, feature_groups=feature_groups, random_state=seed)
    dataloaders = create_dataloaders(data_bundle, batch_size=batch_size)

    model = TabMStyleMultiExpertMLP(
        view_dims=data_bundle.view_dims,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden_dim,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in dataloaders["train"]:
            *views, labels = batch
            views = [tensor.to(device) for tensor in views]
            labels = labels.to(device)

            optimizer.zero_grad()
            expert_logits, _ = model(*views)
            expert_probs = torch.softmax(expert_logits, dim=-1)
            mean_prob = expert_probs.mean(dim=1).clamp_min(1e-8)
            log_mean_prob = torch.log(mean_prob)
            sample_loss = -log_mean_prob.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = (sample_loss * class_weights_t[labels]).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(dataloaders["train"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloaders["val"]:
                *views, labels = batch
                views = [tensor.to(device) for tensor in views]
                labels = labels.to(device)
                expert_logits, _ = model(*views)
                expert_probs = torch.softmax(expert_logits, dim=-1)
                mean_prob = expert_probs.mean(dim=1).clamp_min(1e-8)
                log_mean_prob = torch.log(mean_prob)
                sample_loss = -log_mean_prob.gather(1, labels.unsqueeze(1)).squeeze(1)
                loss = (sample_loss * class_weights_t[labels]).mean()
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_idx, test_prob, test_gates = predict_tabm_style(model, dataloaders["test"], device)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])
    gate_frame = pd.DataFrame(test_gates, columns=list(data_bundle.view_dims.keys()))
    gate_frame["true_label"] = data_bundle.test_labels_text

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "view_dims": data_bundle.view_dims,
        "num_experts": num_experts,
        "expert_hidden_dim": expert_hidden_dim,
        "history": history,
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
        "gate_frame": gate_frame,
    }


def train_ft_transformer(
    train_df,
    test_df,
    seed=RANDOM_STATE,
    d_token=32,
    n_blocks=2,
    attention_n_heads=4,
    attention_dropout=0.2,
    ff_dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=80,
    patience=12,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_ft_transformer_data(train_df, test_df, random_state=seed)
    dataloaders = create_ft_dataloaders(data_bundle, batch_size=batch_size)

    model = FTTransformer(
        cat_cardinalities=data_bundle.cat_cardinalities,
        n_num_features=len(data_bundle.num_feature_names),
        d_token=d_token,
        n_blocks=n_blocks,
        attention_n_heads=attention_n_heads,
        attention_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch_ft(model, dataloaders["train"], criterion, optimizer, device)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, labels in dataloaders["val"]:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                labels = labels.to(device)
                logits = model(x_cat, x_num)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_idx, test_prob = predict_ft_transformer(model, dataloaders["test"], device)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "cat_feature_names": data_bundle.cat_feature_names,
        "num_feature_names": data_bundle.num_feature_names,
        "cat_cardinalities": data_bundle.cat_cardinalities,
        "history": history,
        "config": {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "attention_n_heads": attention_n_heads,
            "attention_dropout": attention_dropout,
            "ff_dropout": ff_dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
    }


def train_piecewise_ft_transformer(
    train_df,
    test_df,
    seed=RANDOM_STATE,
    d_token=48,
    n_blocks=2,
    attention_n_heads=4,
    attention_dropout=0.2,
    ff_dropout=0.2,
    n_bins=8,
    lr=8e-4,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=60,
    patience=10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_ft_transformer_data(train_df, test_df, random_state=seed, n_bins=n_bins)
    dataloaders = create_ft_dataloaders(data_bundle, batch_size=batch_size)

    model = PiecewiseFTTransformer(
        cat_cardinalities=data_bundle.cat_cardinalities,
        numeric_bin_edges=data_bundle.numeric_bin_edges,
        d_token=d_token,
        n_blocks=n_blocks,
        attention_n_heads=attention_n_heads,
        attention_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch_ft(model, dataloaders["train"], criterion, optimizer, device)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, labels in dataloaders["val"]:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                labels = labels.to(device)
                logits = model(x_cat, x_num)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_idx, test_prob = predict_ft_transformer(model, dataloaders["test"], device)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "cat_feature_names": data_bundle.cat_feature_names,
        "num_feature_names": data_bundle.num_feature_names,
        "numeric_bin_edges_shape": None if data_bundle.numeric_bin_edges is None else list(data_bundle.numeric_bin_edges.shape),
        "history": history,
        "config": {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "attention_n_heads": attention_n_heads,
            "attention_dropout": attention_dropout,
            "ff_dropout": ff_dropout,
            "n_bins": n_bins,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
    }


def train_view_aware_ft_transformer(
    train_df,
    test_df,
    seed=RANDOM_STATE,
    d_token=48,
    n_blocks=2,
    attention_n_heads=4,
    attention_dropout=0.2,
    ff_dropout=0.2,
    n_num_frequencies=8,
    lr=8e-4,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=60,
    patience=10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_ft_transformer_data(train_df, test_df, random_state=seed)
    dataloaders = create_ft_dataloaders(data_bundle, batch_size=batch_size)

    model = ViewAwareFTTransformer(
        cat_cardinalities=data_bundle.cat_cardinalities,
        n_num_features=len(data_bundle.num_feature_names),
        cat_view_ids=data_bundle.cat_view_ids,
        num_view_ids=data_bundle.num_view_ids,
        n_views=len(data_bundle.view_names),
        d_token=d_token,
        n_blocks=n_blocks,
        attention_n_heads=attention_n_heads,
        attention_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        n_num_frequencies=n_num_frequencies,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch_ft(model, dataloaders["train"], criterion, optimizer, device)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, labels in dataloaders["val"]:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                labels = labels.to(device)
                logits = model(x_cat, x_num)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_idx, test_prob = predict_ft_transformer(model, dataloaders["test"], device)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "cat_feature_names": data_bundle.cat_feature_names,
        "num_feature_names": data_bundle.num_feature_names,
        "cat_cardinalities": data_bundle.cat_cardinalities,
        "view_names": data_bundle.view_names,
        "history": history,
        "config": {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "attention_n_heads": attention_n_heads,
            "attention_dropout": attention_dropout,
            "ff_dropout": ff_dropout,
            "n_num_frequencies": n_num_frequencies,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
    }


def train_viewwise_gated_ft_transformer(
    train_df,
    test_df,
    seed=RANDOM_STATE,
    d_token=48,
    n_blocks=2,
    attention_n_heads=4,
    attention_dropout=0.2,
    ff_dropout=0.2,
    n_num_frequencies=8,
    lr=8e-4,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=60,
    patience=10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_ft_transformer_data(train_df, test_df, random_state=seed)
    dataloaders = create_ft_dataloaders(data_bundle, batch_size=batch_size)

    model = ViewWiseGatedFTTransformer(
        cat_cardinalities=data_bundle.cat_cardinalities,
        n_num_features=len(data_bundle.num_feature_names),
        cat_view_ids=data_bundle.cat_view_ids,
        num_view_ids=data_bundle.num_view_ids,
        n_views=len(data_bundle.view_names),
        d_token=d_token,
        n_blocks=n_blocks,
        attention_n_heads=attention_n_heads,
        attention_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        n_num_frequencies=n_num_frequencies,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_cat, x_num, labels in dataloaders["train"]:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_cat, x_num)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(dataloaders["train"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, labels in dataloaders["val"]:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                labels = labels.to(device)
                logits, _ = model(x_cat, x_num)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_idx, test_prob, test_gates = predict_gated_ft_transformer(model, dataloaders["test"], device)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])
    gate_frame = pd.DataFrame(test_gates, columns=data_bundle.view_names)
    gate_frame["true_label"] = data_bundle.test_labels_text

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "cat_feature_names": data_bundle.cat_feature_names,
        "num_feature_names": data_bundle.num_feature_names,
        "view_names": data_bundle.view_names,
        "history": history,
        "config": {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "attention_n_heads": attention_n_heads,
            "attention_dropout": attention_dropout,
            "ff_dropout": ff_dropout,
            "n_num_frequencies": n_num_frequencies,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
        "gate_frame": gate_frame,
    }


def train_viewwise_attention_gated_ft_transformer(
    train_df,
    test_df,
    seed=RANDOM_STATE,
    d_token=48,
    n_blocks=2,
    attention_n_heads=4,
    attention_dropout=0.2,
    ff_dropout=0.2,
    n_num_frequencies=8,
    lr=8e-4,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=60,
    patience=10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_ft_transformer_data(train_df, test_df, random_state=seed)
    dataloaders = create_ft_dataloaders(data_bundle, batch_size=batch_size)

    model = ViewWiseAttentionGatedFTTransformer(
        cat_cardinalities=data_bundle.cat_cardinalities,
        n_num_features=len(data_bundle.num_feature_names),
        cat_view_ids=data_bundle.cat_view_ids,
        num_view_ids=data_bundle.num_view_ids,
        n_views=len(data_bundle.view_names),
        d_token=d_token,
        n_blocks=n_blocks,
        attention_n_heads=attention_n_heads,
        attention_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        n_num_frequencies=n_num_frequencies,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_cat, x_num, labels in dataloaders["train"]:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_cat, x_num)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(dataloaders["train"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, labels in dataloaders["val"]:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                labels = labels.to(device)
                logits, _ = model(x_cat, x_num)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_idx, test_prob, test_gates = predict_gated_ft_transformer(model, dataloaders["test"], device)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])
    gate_frame = pd.DataFrame(test_gates, columns=data_bundle.view_names)
    gate_frame["true_label"] = data_bundle.test_labels_text

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "cat_feature_names": data_bundle.cat_feature_names,
        "num_feature_names": data_bundle.num_feature_names,
        "view_names": data_bundle.view_names,
        "history": history,
        "config": {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "attention_n_heads": attention_n_heads,
            "attention_dropout": attention_dropout,
            "ff_dropout": ff_dropout,
            "n_num_frequencies": n_num_frequencies,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
        "gate_frame": gate_frame,
    }


def train_trompt_style_transformer(
    train_df,
    test_df,
    seed=RANDOM_STATE,
    d_token=48,
    n_prompts=8,
    n_blocks=2,
    attention_n_heads=4,
    attention_dropout=0.2,
    ff_dropout=0.2,
    n_num_frequencies=8,
    lr=8e-4,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=60,
    patience=10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_ft_transformer_data(train_df, test_df, random_state=seed)
    dataloaders = create_ft_dataloaders(data_bundle, batch_size=batch_size)

    model = TromptStylePromptTransformer(
        cat_cardinalities=data_bundle.cat_cardinalities,
        n_num_features=len(data_bundle.num_feature_names),
        d_token=d_token,
        n_prompts=n_prompts,
        n_blocks=n_blocks,
        attention_n_heads=attention_n_heads,
        attention_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        n_num_frequencies=n_num_frequencies,
        num_classes=len(LABELS),
    ).to(device)

    class_counts = np.bincount(data_bundle.train_labels_idx, minlength=len(LABELS))
    class_weights = len(data_bundle.train_labels_idx) / (len(LABELS) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_cat, x_num, labels in dataloaders["train"]:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_cat, x_num)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(dataloaders["train"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, labels in dataloaders["val"]:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                labels = labels.to(device)
                logits, _ = model(x_cat, x_num)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(dataloaders["val"].dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probs = []
    preds = []
    importance_frames = []
    with torch.no_grad():
        for x_cat, x_num, _ in dataloaders["test"]:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            logits, importance = model(x_cat, x_num)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            probs.append(prob.cpu().numpy())
            preds.append(pred.cpu().numpy())
            importance_frames.append(importance.cpu().numpy())

    test_prob = np.concatenate(probs, axis=0)
    test_pred_idx = np.concatenate(preds, axis=0)
    test_pred_labels = np.array([LABELS[idx] for idx in test_pred_idx])

    feature_names = data_bundle.cat_feature_names + data_bundle.num_feature_names
    importance_frame = pd.DataFrame(np.concatenate(importance_frames, axis=0), columns=feature_names)
    importance_frame["true_label"] = data_bundle.test_labels_text

    metadata = {
        "device": str(device),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "cat_feature_names": data_bundle.cat_feature_names,
        "num_feature_names": data_bundle.num_feature_names,
        "history": history,
        "config": {
            "d_token": d_token,
            "n_prompts": n_prompts,
            "n_blocks": n_blocks,
            "attention_n_heads": attention_n_heads,
            "attention_dropout": attention_dropout,
            "ff_dropout": ff_dropout,
            "n_num_frequencies": n_num_frequencies,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
    }

    return {
        "model": model,
        "metadata": metadata,
        "test_pred_labels": test_pred_labels,
        "test_prob": test_prob,
        "test_true_labels": data_bundle.test_labels_text,
        "importance_frame": importance_frame,
    }
