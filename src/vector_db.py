import chromadb
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import json

class AnomalyVectorDB:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="system_anomalies",
            metadata={"description": "System monitoring anomalies database"}
        )
    
    def store_anomaly(self, 
                     anomaly_data: Dict[str, Any],
                     timestamp: datetime = None) -> str:
        """Store anomaly data in vector database"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create unique ID
        anomaly_id = f"anomaly_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare document
        document = self._prepare_anomaly_document(anomaly_data, timestamp)
        
        # Store in vector DB
        self.collection.add(
            ids=[anomaly_id],
            documents=[document['text']],
            metadatas=[document['metadata']],
            embeddings=[self._generate_embedding(anomaly_data)]
        )
        
        return anomaly_id
    
    def _prepare_anomaly_document(self, anomaly_data: Dict, timestamp: datetime) -> Dict:
        """Prepare anomaly document for storage"""
        # Create descriptive text
        text_parts = [
            f"System Anomaly Detected at {timestamp}",
            f"Metric: {anomaly_data.get('metric', 'Unknown')}",
            f"Type: {anomaly_data.get('anomaly_type', 'Statistical')}",
            f"Severity: {anomaly_data.get('severity', 'Medium')}",
            f"Value: {anomaly_data.get('actual_value', 'N/A')}",
            f"Expected: {anomaly_data.get('expected_value', 'N/A')}",
            f"Deviation: {anomaly_data.get('deviation', 'N/A')}",
            f"Confidence: {anomaly_data.get('confidence', 'N/A')}"
        ]
        
        if 'additional_context' in anomaly_data:
            text_parts.append(f"Context: {anomaly_data['additional_context']}")
        
        document_text = "\n".join(text_parts)
        
        # Metadata
        metadata = {
            'timestamp': timestamp.isoformat(),
            'metric': anomaly_data.get('metric', ''),
            'anomaly_type': anomaly_data.get('anomaly_type', ''),
            'severity': anomaly_data.get('severity', 'medium'),
            'confidence': anomaly_data.get('confidence', 0.5),
            'actual_value': float(anomaly_data.get('actual_value', 0)),
            'expected_value': float(anomaly_data.get('expected_value', 0)),
            'deviation': float(anomaly_data.get('deviation', 0))
        }
        
        return {
            'text': document_text,
            'metadata': metadata
        }
    
    def _generate_embedding(self, anomaly_data: Dict) -> List[float]:
        """Generate embedding for anomaly data (simplified version)"""
        # In a real implementation, you would use a proper embedding model
        # For now, we'll create a simple numerical representation
        features = [
            anomaly_data.get('severity_score', 0.5),
            anomaly_data.get('deviation', 0),
            anomaly_data.get('confidence', 0.5),
            float(anomaly_data.get('actual_value', 0)) / 1000,  # normalized
        ]
        
        # Pad or truncate to 384 dimensions (common embedding size)
        embedding = features + [0.0] * (384 - len(features))
        return embedding[:384]
    
    def search_anomalies(self, 
                        query: str,
                        n_results: int = 5,
                        filters: Dict = None) -> List[Dict]:
        """Search for similar anomalies"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return formatted_results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Dict]:
        """Get recent anomalies from the database"""
        cutoff_time = (datetime.now() - pd.Timedelta(hours=hours)).isoformat()
        
        filters = {
            'timestamp': {'$gte': cutoff_time}
        }
        
        # Get all results and filter by time
        all_results = self.collection.get()
        recent_anomalies = []
        
        for i, metadata in enumerate(all_results['metadatas']):
            if metadata['timestamp'] >= cutoff_time:
                recent_anomalies.append({
                    'id': all_results['ids'][i],
                    'document': all_results['documents'][i],
                    'metadata': metadata
                })
        
        return recent_anomalies
    
    def get_anomaly_statistics(self) -> Dict:
        """Get statistics about stored anomalies"""
        all_anomalies = self.collection.get()
        
        if not all_anomalies['metadatas']:
            return {}
        
        df = pd.DataFrame(all_anomalies['metadatas'])
        
        stats = {
            'total_anomalies': len(df),
            'by_metric': df['metric'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_type': df['anomaly_type'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'time_range': {
                'earliest': df['timestamp'].min(),
                'latest': df['timestamp'].max()
            }
        }
        
        return stats