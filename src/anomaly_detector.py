import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.anomaly_thresholds = {}
        
    def detect_anomalies_isolation_forest(self, df : pd.DataFrame, 
    contamination : float = 0.1) -> Dict:
        features = self._select_features(df)
        scaled_features = self.scaler.fit_transform(features)
        iso_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100
        )
        anomalies = iso_forest.fit_predict(scaled_features)
        return{
            'anomalies' : anomalies,
            'scores' : iso_forest.decision_function(scaled_features),
            'model' : iso_forest,
            'feature_importance': self._get_feature_importance(iso_forest, features.columns)
        }
    
    def detect_anomalies_dbscan(self, df : pd.DataFrame, eps : float = 0.5,
    min_samples : int = 5) -> Dict:
        features = self._select_features(df)
        X = self.scaler.fit_transform(features)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)
        
        anomalies = np.where(clusters == -1, -1, 1)
        
        return {
            'anomalies': anomalies,
            'clusters': clusters,
            'model': dbscan
        }
    
    def statistical_anomaly_detection(self, df : pd.DataFrame, z_threshold : float = 3.0) -> Dict:
        anomalies_all, z_scores_all = {}, {}
        for col in df.select_dtypes(include = [np.number]).columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            anomalies = (z_scores > z_threshold).astype(int)
            
            anomalies_all[col] = anomalies
            z_scores_all[col] = z_scores
        return {
            'anomalies': anomalies_all,
            'z_scores': z_scores_all,
            'threshold': z_threshold
        }
    
    def multivariate_anomaly_detection(self, df: pd.DataFrame) -> Dict:
        """Multivariate anomaly detection using Mahalanobis distance"""
        # from scipy import stats
        from scipy.spatial import distance
        
        features = self._select_features(df)
        X = self.scaler.fit_transform(features)
        
        # Calculate Mahalanobis distance
        cov_matrix = np.cov(X.T)
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        mean = np.mean(X, axis=0)
        mahalanobis_dist = np.array([
            distance.mahalanobis(row, mean, inv_cov_matrix) 
            for row in X
        ])
        
        # Detect anomalies (beyond 95th percentile)
        threshold = np.percentile(mahalanobis_dist, 95)
        anomalies = (mahalanobis_dist > threshold).astype(int)
        
        return {
            'anomalies': anomalies,
            'distances': mahalanobis_dist,
            'threshold': threshold
        }
        
    def _select_features(self, df : pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols]
    
    def _get_feature_importance(self, model, feature_names) -> Dict:
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
        else:
            importance = {name: 1.0 for name in feature_names}
        
        return importance
    
    def comprehensive_anomaly_detection(self, df : pd.DataFrame) -> Dict:
        results = {}
        
        # Isolation Forest
        results['isolation_forest'] = self.detect_anomalies_isolation_forest(df)
        
        # Statistical detection
        results['statistical'] = self.statistical_anomaly_detection(df)
        
        # Multivariate detection
        results['multivariate'] = self.multivariate_anomaly_detection(df)
        
        # Combine results for consensus
        combined_anomalies = self._combine_anomaly_detectors(results, len(df))
        
        results['combined'] = combined_anomalies
        
        return results
    
    def _combine_anomaly_detectors(self, results: Dict, n_samples: int) -> Dict:
        """Combine results from multiple anomaly detectors"""
        weights = {
            'isolation_forest': 0.4,
            'statistical': 0.3,
            'multivariate': 0.3
        }
        
        combined_score = np.zeros(n_samples)
        
        if_anomalies = results['isolation_forest']['anomalies']
        combined_score += weights['isolation_forest'] * (if_anomalies == -1).astype(float)
        
        if isinstance(results['statistical']['anomalies'], dict):
            results['statistical']['anomalies'] = pd.DataFrame(results['statistical']['anomalies'])
    
        stat_anomalies = results['statistical']['anomalies'].any(axis=1).astype(float)
        combined_score += weights['statistical'] * stat_anomalies
        
        multi_anomalies = results['multivariate']['anomalies']
        combined_score += weights['multivariate'] * multi_anomalies
        
        # Final anomaly decision
        final_anomalies = (combined_score > 0.5).astype(int)
        
        return {
            'anomaly_score': combined_score,
            'anomalies': final_anomalies,
            'confidence': combined_score
        }