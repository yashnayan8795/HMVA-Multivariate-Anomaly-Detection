from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AnomalyEnsemble:
    def __init__(self, pca_var: float = 0.95, iforest_estimators: int = 200, random_state: int = 42):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.pca = PCA(n_components=pca_var, svd_solver='full', random_state=random_state)
        self.iforest = IsolationForest(
            n_estimators=iforest_estimators,
            contamination='auto',
            bootstrap=False,
            random_state=random_state,
            n_jobs=-1
        )
        self.colnames: List[str] = []
        self.train_means: np.ndarray | None = None
        self.fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> None:
        self.colnames = list(train_df.columns)
        X = train_df.to_numpy(dtype=float)
        # standardize
        Z = self.scaler.fit_transform(X)
        self.train_means = self.scaler.mean_
        # PCA
        self.pca.fit(Z)
        # Isolation Forest on standardized space
        self.iforest.fit(Z)
        self.fitted = True

    def _ensure_fitted(self):
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def _pca_reconstruction_error(self, Z: np.ndarray) -> np.ndarray:
        # Reconstruct in standardized space
        Z_hat = self.pca.inverse_transform(self.pca.transform(Z))
        # per-feature absolute reconstruction error
        recon = np.abs(Z - Z_hat)
        return recon

    def score_rows(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self._ensure_fitted()
        # align and transform
        X = df[self.colnames].to_numpy(dtype=float)
        Z = self.scaler.transform(X)

        # Channel S1: univariate z-score magnitude (L2 norm of z)
        S1 = np.linalg.norm(Z, axis=1)

        # Channel S2: PCA reconstruction error magnitude (L2 norm of per-feature abs error)
        recon = self._pca_reconstruction_error(Z)
        S2 = np.linalg.norm(recon, axis=1)

        # Channel S3: IsolationForest anomaly score (negative decision function => higher is more anomalous)
        # sklearn decision_function: higher is "more normal"; so use negative
        decf = self.iforest.decision_function(Z)
        S3 = -decf

        return {"S1": S1, "S2": S2, "S3": S3, "Z": Z, "RECON": recon, "DECF": decf}

    def per_row_contributions(self, Z_row: np.ndarray, recon_row: np.ndarray) -> Dict[str, np.ndarray]:
        # Base contributions from |z| and |recon| per feature (standardized space)
        z_contrib = np.abs(Z_row)
        pca_contrib = np.abs(recon_row)

        # IsolationForest sensitivity via mean-imputation delta on this row
        base_df = Z_row.reshape(1, -1)
        base_s3 = -self.iforest.decision_function(base_df)[0]  # higher = more anomalous

        sens = np.zeros_like(Z_row)
        for i in range(len(Z_row)):
            z_mod = Z_row.copy()
            z_mod[i] = 0.0  # replace with mean in standardized space
            s3_mod = -self.iforest.decision_function(z_mod.reshape(1, -1))[0]
            delta = max(0.0, base_s3 - s3_mod)  # only positive reductions
            sens[i] = delta

        # Combine contributions (sum of channels). We will normalize later.
        contrib = z_contrib + pca_contrib + sens
        return {"z": z_contrib, "pca": pca_contrib, "if_delta": sens, "total": contrib}
    
    def if_sensitivity_matrix(self, Z: np.ndarray) -> np.ndarray:
        """
            Compute IsolationForest sensitivity for ALL rows in batch:
            For each feature i, set Z[:, i]=0 (mean in standardized space) and
            measure the reduction in anomaly score vs the base.
            Returns: (n_rows, n_features) matrix of positive deltas.
        """
        self._ensure_fitted()
        # base anomaly score (higher => more anomalous)
        base = -self.iforest.decision_function(Z)  # shape (n_rows,)
        n_rows, n_feats = Z.shape
        deltas = np.zeros((n_rows, n_feats), dtype=float)

        # Do one full pass per feature (n_feats IF predictions total)
        for i in range(n_feats):
            Z_mod = Z.copy()
            Z_mod[:, i] = 0.0  # replace with mean in standardized space
            mod = -self.iforest.decision_function(Z_mod)  # shape (n_rows,)
            # only positive reductions count as "explanation"
            deltas[:, i] = np.maximum(0.0, base - mod)

        return deltas

