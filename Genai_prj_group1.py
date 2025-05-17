import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import requests
from bs4 import BeautifulSoup
from groq import Groq
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from transformers import pipeline
import re
from urllib.parse import urlparse
from datetime import datetime
import time
from serpapi import GoogleSearch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Download NLTK resources
nltk.download('vader_lexicon')

# Configure app
st.set_page_config(
    page_title="AI FactGuard PRO",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        background-color: #0062cc;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        margin-bottom: 20px;
        background-color: #ffffff;
    }
    .verdict-true {
        color: #28a745;
        font-weight: bold;
    }
    .verdict-false {
        color: #dc3545;
        font-weight: bold;
    }
    .verdict-misleading {
        color: #ffc107;
        font-weight: bold;
    }
    .verdict-unverifiable {
        color: #6c757d;
        font-weight: bold;
    }
    .tab-content {
        padding-top: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #0062cc;
    }
    .stButton>button {
        background-color: #0062cc;
        color: white;
    }
    .search-result {
        border-left: 3px solid #0062cc;
        padding-left: 10px;
        margin-bottom: 10px;
    }
    .claim-box {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .model-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
    }
    .model-comparison {
        background-color: #f1f3f5;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #0062cc;
    }
</style>
""", unsafe_allow_html=True)

class AITools:
    def __init__(self):
        # Initialize Groq client
        self.groq_client = Groq(api_key="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")
        
        # Initialize SerpAPI
        self.serpapi_key = "439df6edea5a0ead58e65a5a1cba469c76b8161aed8d7087f20a893e497ae5c9"
        
        # Available models with descriptions and performance metrics
        self.available_models = {
            "Llama3-70B": {
                "model_name": "llama3-70b-8192",
                "description": "Meta's most capable openly available LLM to date, optimized for complex reasoning",
                "speed": "Fast",
                "accuracy": "High",
                "context_window": 8192,
                "best_for": "Complex analysis, detailed reports",
                "color": "#636EFA"
            },
            "Llama3-8B": {
                "model_name": "llama3-8b-8192",
                "description": "Efficient smaller version of Llama 3, good balance of speed and capability",
                "speed": "Very Fast",
                "accuracy": "Medium-High",
                "context_window": 8192,
                "best_for": "General purpose, quick analysis",
                "color": "#EF553B"
            },
            "Mixtral-8x7B": {
                "model_name": "mixtral-8x7b-32768",
                "description": "Sparse Mixture of Experts model with excellent cost/performance ratio",
                "speed": "Medium",
                "accuracy": "Very High",
                "context_window": 32768,
                "best_for": "Large context, specialized tasks",
                "color": "#00CC96"
            }
        }
        
        # Initialize local models
        try:
            self.bias_analyzer = pipeline(
                "text-classification", 
                model="valurank/distilroberta-bias",
                device="cpu"
            )
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            st.error(f"Failed to initialize local models: {str(e)}")
            self.bias_analyzer = None
            self.sentiment_analyzer = None
        
        # Track model usage for comparison
        self.model_usage_stats = defaultdict(int)
        self.model_response_times = defaultdict(list)
        self.model_verdicts = defaultdict(lambda: defaultdict(int))

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def fetch_url_content(self, url: str) -> str:
        """Fetch and clean content from URL with robust error handling"""
        if not self._is_valid_url(url):
            st.error("❌ Invalid URL format. Please include http:// or https://")
            return None

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            
            with st.spinner(f"Fetching content from {url[:50]}..."):
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()

                # Detect if content is paywalled
                if "paywall" in response.text.lower() or "subscribe" in response.text.lower():
                    st.warning("⚠️ Paywall detected. Content may be incomplete.")

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 
                                   'noscript', 'header', 'aside', 'form', 'button']):
                    element.decompose()
                
                # Extract and clean text
                paragraphs = []
                for p in soup.find_all('p'):
                    text = re.sub(r'\s+', ' ', p.get_text()).strip()
                    if len(text.split()) > 7:  # Only keep meaningful paragraphs
                        paragraphs.append(text)
                
                content = ' '.join(paragraphs)
                
                if not content:
                    st.error("No meaningful content found. The site structure may not be supported.")
                    return None
                
                return content[:20000]  # Limit to 20k characters
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to fetch URL: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return None

    def search_web(self, query: str, num_results: int = 5) -> list:
        """Perform real-time web search using SerpAPI"""
        try:
            params = {
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results,
                "hl": "en",
                "gl": "us"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Filter for reliable sources
            reliable_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 
                              'washingtonpost.com', 'theguardian.com', 'wsj.com']
            filtered_results = [
                r for r in results.get("organic_results", [])
                if any(domain in r.get('link', '') for domain in reliable_domains)
            ]
            
            return filtered_results[:num_results]  # Return top N results
        
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            return []

    def analyze_sentiment(self, text: str) -> tuple:
        """Analyze text sentiment with VADER"""
        if not text or not self.sentiment_analyzer:
            return "Error", 0.0
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return "Positive", scores['compound']
        elif scores['compound'] <= -0.05:
            return "Negative", scores['compound']
        else:
            return "Neutral", scores['compound']

    def analyze_bias(self, text: str) -> dict:
        """Analyze political/media bias"""
        if not text or not self.bias_analyzer:
            return {"label": "Error", "score": 0.0, "explanation": "Analysis unavailable"}
        
        try:
            result = self.bias_analyzer(text[:2000])  # Limit input size
            explanation = self._generate_bias_explanation(text[:1000], result[0]['label'])
            return {
                "label": result[0]['label'],
                "score": result[0]['score'],
                "explanation": explanation
            }
        except Exception as e:
            return {"label": "Error", "score": 0.0, "explanation": f"Analysis failed: {str(e)}"}

    def _generate_bias_explanation(self, text: str, label: str) -> str:
        """Generate explanation for bias analysis"""
        prompt = f"""Explain why this text might be considered {label} biased in one paragraph.
        Highlight specific words or phrases that contribute to this perception:
        
        {text[:1000]}"""
        
        return self.get_model_response(prompt, model_key="Llama3-8B")

    def extract_claims(self, text: str) -> list:
        """Extract factual claims from text with improved reliability"""
        prompt = f"""Analyze the following text and extract 3-5 specific factual claims that can be verified. 
        Format each claim exactly as follows (include the quotes):
        - "Claim 1 text"
        - "Claim 2 text"
        - "Claim 3 text"
        
        Focus on claims that make specific assertions about facts, statistics, events, or statements that could be 
        objectively verified. Skip opinions, predictions, and vague statements.
        
        Text:
        {text[:3000]}"""
        
        response = self.get_model_response(prompt, model_key="Llama3-8B")
        
        # Improved claim extraction with regex
        claims = []
        for line in response.split('\n'):
            match = re.match(r'-\s*"(.*?)"', line.strip())
            if match:
                claims.append(match.group(1))
        
        if not claims:
            return ["No verifiable factual claims found in the text"]
        
        return claims[:5]  # Return max 5 claims

    def verify_claim(self, claim: str, model_key: str = "Llama3-70B") -> dict:
        """Enhanced claim verification with web search and improved parsing"""
        # First get LLM analysis with more structured prompt
        prompt = f"""Analyze the following claim and provide a detailed verification. Use EXACTLY this format:
        
        VERDICT: [True/False/Misleading/Unverifiable]
        CONFIDENCE: [0-100%]
        REASONING: [2-3 sentence explanation of the verdict]
        
        Be objective and thorough. If the claim cannot be verified with high confidence, 
        mark it as Unverifiable rather than guessing.
        
        Claim: "{claim}" """
        
        llm_response = self._parse_verification_response(
            self.get_model_response(prompt, model_key=model_key))
        
        # Track verdict for model comparison
        self.model_verdicts[model_key][llm_response["verdict"]] += 1
        
        # Perform web search for additional verification
        search_results = self.search_web(claim)
        
        # Add search results to response
        llm_response["sources"] = search_results
        
        # Cross-validate with web results if available
        if search_results:
            supporting = 0
            contradicting = 0
            neutral = 0
            
            for result in search_results:
                snippet = result.get('snippet', '').lower()
                if any(term in snippet for term in ['not true', 'false', 'debunked', 'misleading', 'inaccurate']):
                    contradicting += 1
                elif any(term in snippet for term in ['true', 'accurate', 'confirmed', 'correct']):
                    supporting += 1
                else:
                    neutral += 1
            
            # Adjust confidence based on web results
            if supporting > contradicting and llm_response["verdict"] == "True":
                llm_response["confidence"] = min(100, llm_response["confidence"] + (supporting * 5))
            elif contradicting > supporting and llm_response["verdict"] == "False":
                llm_response["confidence"] = min(100, llm_response["confidence"] + (contradicting * 5))
            elif neutral > (supporting + contradicting):
                llm_response["confidence"] = max(0, llm_response["confidence"] - 10)
        
        return llm_response

    def _parse_verification_response(self, response: str) -> dict:
        """Improved parsing of the structured verification response"""
        result = {
            "verdict": "Unverifiable",
            "confidence": 50,  # Default to 50% if not specified
            "reasoning": "Could not determine veracity with confidence",
            "sources": []
        }
        
        # Extract verdict
        verdict_match = re.search(r'VERDICT:\s*(True|False|Misleading|Unverifiable)', response, re.IGNORECASE)
        if verdict_match:
            result["verdict"] = verdict_match.group(1).capitalize()
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)%?', response)
        if confidence_match:
            try:
                result["confidence"] = min(100, max(0, int(confidence_match.group(1))))
            except:
                pass
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\n\n|\nVERDICT:|$)', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # Fallback if standard format not found
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            if len(lines) >= 3:
                result["reasoning"] = '\n'.join(lines[2:])
        
        return result

    def get_model_response(self, prompt: str, model_key: str = "Llama3-70B", temp: float = 0.3) -> str:
        """Get response from Groq model with error handling"""
        try:
            start_time = time.time()
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.available_models[model_key]["model_name"],
                temperature=temp,
                max_tokens=1024
            )
            end_time = time.time()
            
            # Track model usage and response time
            self.model_usage_stats[model_key] += 1
            self.model_response_times[model_key].append(end_time - start_time)
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"AI service error: {str(e)}")
            return f"AI analysis failed: {str(e)}"

    def generate_report(self, content: str) -> dict:
        """Generate comprehensive analysis report with improved claim handling"""
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content_preview": content[:500] + "..." if len(content) > 500 else content,
            "content_length": len(content)
        }
        
        # Sentiment analysis
        report["sentiment"], report["sentiment_score"] = self.analyze_sentiment(content)
        
        # Bias analysis
        report["bias"] = self.analyze_bias(content)
        
        # Claim verification (with progress indicator)
        with st.spinner("Extracting and verifying claims..."):
            claims = self.extract_claims(content)
            report["claims"] = []
            
            progress_bar = st.progress(0)
            total_claims = min(5, len(claims))  # Verify max 5 claims
            
            for i, claim in enumerate(claims[:5], 1):  # Limit to 5 claims
                report["claims"].append({
                    "text": claim,
                    "analysis": self.verify_claim(claim, model_key=st.session_state.current_model)
                })
                progress_bar.progress(i / total_claims)
                time.sleep(0.1)  # Small delay for progress bar visibility
        
        # Generate summary
        report["summary"] = self._generate_report_summary(report)
        
        return report

    def _generate_report_summary(self, report: dict) -> str:
        """Generate human-readable report summary with claim highlights"""
        prompt = f"""Create a concise 3-paragraph news analysis report with:
        1. Overall sentiment and bias assessment
        2. Key claim veracity highlights (mention any False/Misleading claims first)
        3. Reliability conclusion and recommendations
        
        Sentiment: {report['sentiment']} (score: {report['sentiment_score']:.2f})
        Bias: {report['bias']['label']} (confidence: {report['bias']['score']:.0%})
        Claims analyzed: {len(report['claims'])}
        
        Claim Verdicts:
        {', '.join(f"{i+1}: {c['analysis']['verdict']}" for i, c in enumerate(report['claims']))}"""
        
        return self.get_model_response(prompt, model_key="Llama3-70B")

    def get_model_comparison_data(self):
        """Prepare data for model comparison visualizations"""
        # Prepare usage data
        usage_data = []
        for model, count in self.model_usage_stats.items():
            usage_data.append({
                "Model": model,
                "Usage Count": count,
                "Description": self.available_models[model]["description"],
                "Average Response Time": sum(self.model_response_times[model])/len(self.model_response_times[model]) if model in self.model_response_times and self.model_response_times[model] else 0,
                "Color": self.available_models[model]["color"]
            })
        
        # Prepare performance data
        performance_data = []
        for model in self.available_models:
            perf = {
                "Model": model,
                "Speed": self.available_models[model]["speed"],
                "Accuracy": self.available_models[model]["accuracy"],
                "Context Window": self.available_models[model]["context_window"],
                "Best For": self.available_models[model]["best_for"],
                "Color": self.available_models[model]["color"]
            }
            
            # Add actual performance metrics if available
            if model in self.model_response_times and self.model_response_times[model]:
                perf["Avg Response Time"] = sum(self.model_response_times[model])/len(self.model_response_times[model])
            
            performance_data.append(perf)
        
        # Prepare verdict data
        verdict_data = []
        for model in self.model_verdicts:
            for verdict, count in self.model_verdicts[model].items():
                verdict_data.append({
                    "Model": model,
                    "Verdict": verdict,
                    "Count": count,
                    "Color": self.available_models[model]["color"]
                })
        
        return pd.DataFrame(usage_data), pd.DataFrame(performance_data), pd.DataFrame(verdict_data)

def display_verdict(verdict: str) -> None:
    """Display verdict with appropriate styling"""
    verdict_lower = verdict.lower()
    if "true" in verdict_lower:
        st.markdown(f"<p class='verdict-true'>Verdict: {verdict}</p>", unsafe_allow_html=True)
    elif "false" in verdict_lower:
        st.markdown(f"<p class='verdict-false'>Verdict: {verdict}</p>", unsafe_allow_html=True)
    elif "misleading" in verdict_lower:
        st.markdown(f"<p class='verdict-misleading'>Verdict: {verdict}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='verdict-unverifiable'>Verdict: {verdict}</p>", unsafe_allow_html=True)

def display_search_results(results: list) -> None:
    """Display web search results in Streamlit"""
    if not results:
        st.warning("No relevant web results found")
        return
    
    st.markdown("### 🔍 Web Verification Results")
    for i, result in enumerate(results, 1):
        with st.container():
            st.markdown(f"""
            <div class='search-result'>
                <b>{i}. <a href="{result['link']}" target="_blank">{result.get('title', 'No title')}</a></b>
                <p>{result.get('snippet', 'No preview available')}</p>
                <small>{result['link']}</small>
            </div>
            """, unsafe_allow_html=True)

def display_model_comparison(usage_df, performance_df, verdict_df):
    """Display model comparison visualizations"""
    st.markdown("### Model Performance Comparison")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Usage Statistics", "Performance Metrics", "Verdict Analysis", "Model Specifications"])
    
    with tab1:
        if not usage_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Usage count bar chart
                fig = px.bar(usage_df, x="Model", y="Usage Count", 
                             title="Model Usage Count", color="Model",
                             text="Usage Count",
                             color_discrete_map={row["Model"]: row["Color"] for _, row in usage_df.iterrows()})
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Response time scatter plot
                if "Average Response Time" in usage_df.columns:
                    fig = px.scatter(usage_df, x="Model", y="Average Response Time",
                                    title="Average Response Time (seconds)", 
                                    size="Usage Count", color="Model",
                                    color_discrete_map={row["Model"]: row["Color"] for _, row in usage_df.iterrows()})
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model usage data available yet")
    
    with tab2:
        if not performance_df.empty:
            # Radar chart for performance metrics
            fig = go.Figure()
            
            # Convert speed to numerical values for visualization
            speed_map = {"Very Fast": 4, "Fast": 3, "Medium": 2, "Slow": 1}
            accuracy_map = {"Very High": 4, "High": 3, "Medium-High": 2.5, "Medium": 2, "Low": 1}
            
            for _, row in performance_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[
                        speed_map.get(row["Speed"], 0),
                        accuracy_map.get(row["Accuracy"], 0),
                        row["Context Window"] / 8000,  # Normalize
                        1 if "Avg Response Time" not in row else 5 - min(4, row["Avg Response Time"] * 10)
                    ],
                    theta=['Speed', 'Accuracy', 'Context Window', 'Response Time'],
                    fill='toself',
                    name=row["Model"],
                    line_color=row["Color"]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 4]
                    )),
                showlegend=True,
                title="Model Performance Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.dataframe(performance_df.set_index("Model"), use_container_width=True)
    
    with tab3:
        if not verdict_df.empty:
            col1, col2 = st.columns(2)
        
            with col1:
                # Verdict distribution by model
                fig = px.bar(verdict_df, x="Model", y="Count", color="Verdict",
                             title="Verdict Distribution by Model",
                             barmode="group",
                             color_discrete_map={
                                 'True': '#28a745',
                                'False': '#dc3545',
                                'Misleading': '#ffc107',
                                'Unverifiable': '#6c757d'
                            })
                st.plotly_chart(fig, use_container_width=True)
        
            with col2:
                # Get unique verdicts present in the data
                unique_verdicts = verdict_df['Verdict'].unique()
            
                # Create color mapping only for present verdicts
                color_map = {
                    'True': '#28a745',
                    'False': '#dc3545',
                    'Misleading': '#ffc107',
                    'Unverifiable': '#6c757d'
                }
                present_colors = {k: color_map[k] for k in unique_verdicts if k in color_map}
            
                # Pivot and calculate percentages
                verdict_pivot = verdict_df.pivot_table(
                    index="Model", 
                    columns="Verdict", 
                    values="Count", 
                    aggfunc="sum", 
                    fill_value=0
                )
            
                # Calculate percentages
                verdict_pivot = verdict_pivot.div(verdict_pivot.sum(axis=1), axis=0) * 100
            
                # Melt the DataFrame for plotting
                verdict_percent = verdict_pivot.reset_index().melt(
                    id_vars="Model", 
                    value_vars=list(unique_verdicts),
                    var_name="Verdict",
                    value_name="Percentage"
                )
            
                fig = px.bar(
                    verdict_percent, 
                    x="Model", 
                    y="Percentage", 
                    color="Verdict",
                    title="Verdict Percentage by Model",
                    color_discrete_map=present_colors
                )
                fig.update_layout(barmode="stack")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No verdict data available yet")    
    with tab4:
        st.markdown("### Available Models")
        
        for model_name, model_info in st.session_state.ai.available_models.items():
            with st.expander(f"🔹 {model_name}", expanded=model_name=="Llama3-70B"):
                st.markdown(f"""
                <div class='model-card'>
                    <p><strong>Description:</strong> {model_info['description']}</p>
                    <p><strong>Best For:</strong> {model_info['best_for']}</p>
                    <div class='metric-card'>
                        <p><strong>Speed:</strong> {model_info['speed']}</p>
                        <p><strong>Accuracy:</strong> {model_info['accuracy']}</p>
                    </div>
                    <div class='metric-card'>
                        <p><strong>Context Window:</strong> {model_info['context_window']:,} tokens</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def analysis_dashboard(report: dict) -> None:
    """Display comprehensive analysis results with improved claim verification section"""
    st.subheader("Analysis Results")
    
    # Summary Card
    with st.container():
        st.markdown(f"<div class='card'>{report['summary']}</div>", unsafe_allow_html=True)
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment & Bias", "Claim Verification", "Full Report", "Model Analytics"])
    
    with tab1:
        st.markdown("### Sentiment Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", report["sentiment"])
        with col2:
            st.metric("Sentiment Score", f"{report['sentiment_score']:.2f}")
        
        # Sentiment visualization
        sentiment_score = report['sentiment_score']
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Intensity"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightgray"},
                    {'range': [0.1, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Bias Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Bias", report["bias"]["label"])
        with col2:
            st.metric("Confidence", f"{report['bias']['score']:.0%}")
        
        # Bias visualization
        bias_score = report['bias']['score']
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bias_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Bias Confidence"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "lightyellow"},
                    {'range': [0.7, 1], 'color': "lightcoral"}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Explanation:")
        st.write(report["bias"]["explanation"])
    
    with tab2:
        st.markdown("### Claim Verification")
        
        if not report.get("claims") or report["claims"][0]["text"] == "No verifiable factual claims found in the text":
            st.warning("No verifiable factual claims were found in the content")
        else:
            st.info(f"Analyzed {len(report['claims'])} key claims from the content")
            
            # Prepare data for claim visualization
            claim_data = []
            for i, claim in enumerate(report["claims"], 1):
                if "analysis" in claim:
                    claim_data.append({
                        "Claim": f"Claim {i}",
                        "Verdict": claim["analysis"]["verdict"],
                        "Confidence": claim["analysis"]["confidence"]
                    })
            
            if claim_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create verdict distribution pie chart
                    verdict_df = pd.DataFrame(claim_data)
                    fig = px.pie(verdict_df, names="Verdict", title="Claim Verdict Distribution",
                                 color="Verdict",
                                 color_discrete_map={
                                     'True': '#28a745',
                                     'False': '#dc3545',
                                     'Misleading': '#ffc107',
                                     'Unverifiable': '#6c757d'
                                 })
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create confidence box plot
                    fig = px.box(verdict_df, x="Verdict", y="Confidence", 
                                 title="Confidence Distribution by Verdict",
                                 color="Verdict",
                                 color_discrete_map={
                                     'True': '#28a745',
                                     'False': '#dc3545',
                                     'Misleading': '#ffc107',
                                     'Unverifiable': '#6c757d'
                                 })
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            
            for i, claim in enumerate(report["claims"], 1):
                with st.container():
                    st.markdown(f"""
                    <div class='claim-box'>
                        <h4>Claim {i}:</h4>
                        <p>{claim['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "analysis" in claim:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            display_verdict(claim["analysis"]["verdict"])
                            st.progress(claim["analysis"]["confidence"]/100)
                            st.caption(f"Confidence: {claim['analysis']['confidence']}%")
                        
                        with col2:
                            st.markdown("**Analysis:**")
                            st.write(claim["analysis"]["reasoning"])
                        
                        # Display web search results if available
                        if claim["analysis"].get("sources"):
                            display_search_results(claim["analysis"]["sources"])
                    else:
                        st.warning("Analysis not available for this claim")
                
                st.markdown("---")
    
    with tab3:
        st.markdown("### Complete Technical Report")
        st.json(report, expanded=False)
        
        # Download report
        report_str = f"AI FactGuard Report\nGenerated: {report.get('timestamp', 'N/A')}\n\n"
        report_str += f"Content Length: {report.get('content_length', 0)} characters\n\n"
        report_str += f"SUMMARY:\n{report.get('summary', 'N/A')}\n\n"
        report_str += f"SENTIMENT: {report.get('sentiment', 'N/A')} (Score: {report.get('sentiment_score', 0):.2f})\n"
        report_str += f"BIAS: {report.get('bias', {}).get('label', 'N/A')} (Confidence: {report.get('bias', {}).get('score', 0):.0%})\n\n"
        report_str += "CLAIM VERIFICATIONS:\n"
        
        for i, claim in enumerate(report.get("claims", []), 1):
            report_str += f"\nCLAIM {i}: {claim.get('text', 'N/A')}\n"
            if "analysis" in claim:
                report_str += f"Verdict: {claim.get('analysis', {}).get('verdict', 'N/A')}\n"
                report_str += f"Confidence: {claim.get('analysis', {}).get('confidence', 0)}%\n"
                report_str += f"Analysis: {claim.get('analysis', {}).get('reasoning', 'N/A')}\n"
                report_str += "Sources:\n"
                for j, source in enumerate(claim.get('analysis', {}).get('sources', []), 1):
                    report_str += f"  {j}. {source.get('title', 'No title')}\n"
                    report_str += f"     {source.get('link', 'No URL')}\n"
            else:
                report_str += "Analysis not available for this claim\n"
        
        st.download_button(
            label="📥 Download Full Report",
            data=report_str,
            file_name=f"factguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with tab4:
        # Get model comparison data
        usage_df, performance_df, verdict_df = st.session_state.ai.get_model_comparison_data()
        
        # Display model comparison visualizations
        display_model_comparison(usage_df, performance_df, verdict_df)

def input_section() -> str:
    """Handle user input with improved URL handling"""
    st.subheader("Input News Content")
    input_method = st.radio(
        "Select input method:",
        ("Text", "URL"),
        horizontal=True,
        key="input_method_selector"
    )
    
    content = ""
    if input_method == "URL":
        url = st.text_input(
            "Enter news article URL:",
            placeholder="https://www.bbc.com/news/world-us-canada-12345678",
            key="url_input"
        )
        
        if st.button("Fetch and Analyze", key="fetch_button"):
            if url:
                with st.spinner("Fetching and processing article..."):
                    content = st.session_state.ai.fetch_url_content(url)
                    if content:
                        st.session_state.content = content
                        st.success("✅ Content loaded successfully!")
                        st.session_state.auto_run_analysis = True
                    else:
                        st.session_state.auto_run_analysis = False
            else:
                st.warning("Please enter a URL")
    else:
        content = st.text_area(
            "Paste news content here:",
            height=200,
            key="text_input",
            help="Paste at least 2-3 paragraphs for meaningful analysis"
        )
        if content:
            st.session_state.content = content
    
    return st.session_state.get("content", "")

def show_header():
    """Display application header"""
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; color:white;">AI FactGuard PRO</h1>
        <p style="margin:0; color:white;">Advanced News Verification System</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    show_header()
    
    # Initialize AI tools
    if "ai" not in st.session_state:
        st.session_state.ai = AITools()
    
    # Get user input
    content = input_section()
    
    # Run analysis if conditions met
    if content and (st.session_state.get("auto_run_analysis", False) or 
                    st.button("Run Comprehensive Analysis", type="primary")):
        with st.spinner("Generating comprehensive analysis..."):
            report = st.session_state.ai.generate_report(content)
            analysis_dashboard(report)
            st.session_state.auto_run_analysis = False  # Reset for next run
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### About AI FactGuard PRO")
        st.write("""
        This system analyzes news content using:
        - Groq's ultra-fast LLMs (Llama 3, Mixtral)
        - Real-time web verification (SerpAPI)
        - Sentiment analysis (VADER)
        - Bias detection (distilroberta-bias)
        """)
        
        st.markdown("### Selected Model")
        st.session_state.current_model = st.selectbox(
            "Primary analysis model:",
            list(st.session_state.ai.available_models.keys()),
            index=0
        )
        
        # Display model info card in sidebar
        selected_model_info = st.session_state.ai.available_models[st.session_state.current_model]
        st.markdown(f"""
        <div class='model-card'>
            <h4>{st.session_state.current_model}</h4>
            <p><strong>Speed:</strong> {selected_model_info['speed']}</p>
            <p><strong>Accuracy:</strong> {selected_model_info['accuracy']}</p>
            <p><strong>Context Window:</strong> {selected_model_info['context_window']:,} tokens</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Enter URL or paste text
        2. Click 'Run Comprehensive Analysis'
        3. Explore results in tabs
        """)
        
        if st.session_state.get("content"):
            st.markdown("---")
            st.markdown(f"**Content loaded:** {len(st.session_state['content'])} characters")

if __name__ == "__main__":
    # Clear auto-run flag on start
    if "auto_run_analysis" not in st.session_state:
        st.session_state.auto_run_analysis = False
    
    main()