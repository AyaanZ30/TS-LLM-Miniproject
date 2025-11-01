import google.generativeai as genai
import os
from typing import List, Dict, Any
import json
import pandas as pd

class LLMAnalyzer:
    def __init__(self, api_key : str = None):
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            self.available = True
        else:
            self.available = False
            print("Warning: Google API key not provided. LLM features disabled.")
        
    def analyze_anomaly_patterns(self, anomalic_data : List[Dict]) -> str:
        if not self.available:
            return "LLM analysis unavailable - API key not provided"
        
        try:
            prompt = self._create_analysis_prompt(anomalic_data)
            response = self.model.generate_content(prompt)
            # print(response)
            return response.text
        except Exception as e:
            return f"LLM analysis failed: {str(e)}"
    
    def generate_insights(self, system_metrics: pd.DataFrame, 
                        anomalies: List[Dict]) -> str:
        """Generate insights about system behavior"""
        if not self.available:
            return "LLM insights unavailable - API key not provided"
        
        try:
            prompt = self._create_insights_prompt(system_metrics, anomalies)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Insights generation failed: {str(e)}"
    
    def answer_question(self, question: str, context: List[Dict]) -> str:
        """Answer questions about system anomalies using RAG"""
        if not self.available:
            return "Question answering unavailable - API key not provided"
        
        try:
            prompt = self._create_qa_prompt(question, context)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Question answering failed: {str(e)}"
    
    def _create_analysis_prompt(self, anomalies_data: List[Dict]) -> str:
        """Create prompt for anomaly pattern analysis"""
        anomalies_summary = "\n".join([
            f"- {anom['metadata'].get('metric', 'Unknown')}: "
            f"Severity: {anom['metadata'].get('severity', 'Unknown')}, "
            f"Type: {anom['metadata'].get('anomaly_type', 'Unknown')}, "
            f"Confidence: {anom['metadata'].get('confidence', 0):.2f}"
            for anom in anomalies_data[:10]  # Limit to first 10 for brevity
        ])
        
        prompt = f"""
        Analyze the following system anomalies and provide insights:

        Anomalies Summary:
        {anomalies_summary}

        Please provide:
        1. Common patterns or trends in these anomalies
        2. Potential root causes based on the metrics involved
        3. Recommendations for addressing these issues
        4. Any correlations between different metric anomalies

        Be concise but thorough in your analysis.
        """
        return prompt
    
    def _create_insights_prompt(self, system_metrics: pd.DataFrame, 
                              anomalies: List[Dict]) -> str:
        """Create prompt for system insights generation"""
        metrics_summary = system_metrics.describe().to_string()
        
        prompt = f"""
        Analyze the following system metrics and provide operational insights:

        System Metrics Summary:
        {metrics_summary}

        Key Anomalies Detected: {len(anomalies)}

        Please provide:
        1. Overall system health assessment
        2. Performance bottlenecks identification
        3. Resource utilization recommendations
        4. Predictive maintenance suggestions
        5. Optimization opportunities

        Focus on actionable insights for system administrators.
        """
        
        return prompt

    def _create_qa_prompt(self, question : str, context : List[Dict]) -> str:
        """Create prompt for question answering"""
        context_text = '\n\n'.join([
            f"Anomaly : {i+1}:\n{anom['document']}"
            for i, anom in enumerate(context)
        ])
        prompt = f"""
        Based on the following system anomaly records, answer the user's question.

        Anomaly Records:
        {context_text}

        User Question: {question}

        Instructions:
        - Answer based only on the provided anomaly records
        - Be specific and reference actual metrics and values when possible
        - If the information is not available in the records, say so
        - Provide actionable recommendations when appropriate

        Answer:
        """
        
        return prompt
        