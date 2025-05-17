AI FactGuard PRO is an advanced news verification and fact-checking system that leverages cutting-edge AI models to analyze news content for accuracy, bias, and reliability. The tool helps users determine the credibility of news articles by:

🔍 Key Features
Automated Fact-Checking

Extracts factual claims from news content

Verifies claims against trusted sources using real-time web search

Provides verdicts: True, False, Misleading, or Unverifiable

Sentiment & Bias Analysis

Detects emotional tone (Positive/Negative/Neutral) using VADER sentiment analysis

Identifies political/media bias using a fine-tuned DistilRoBERTa model

Multi-Model AI Verification

Uses Groq's ultra-fast LLMs (Llama3-70B, Llama3-8B, Mixtral-8x7B) for claim verification

Compares model performance with interactive visualizations

Interactive Dashboard

Visual analytics (charts, gauges, and progress bars)

Model comparison (speed, accuracy, verdict distribution)

Downloadable reports (detailed JSON & text summaries)

User-Friendly Interface

Supports URL input (auto-fetch content) or direct text paste

Real-time processing with progress tracking

🛠️ Technical Stack
Backend: Python (Streamlit, Groq API, SerpAPI)

AI Models:

LLMs: Llama3-70B, Llama3-8B, Mixtral-8x7B (via Groq)

Sentiment Analysis: NLTK VADER

Bias Detection: Hugging Face (DistilRoBERTa)

Visualization: Plotly, Pandas

Web Search: SerpAPI (Google Search results)

🎯 Use Cases
✔ Journalists & Researchers – Quickly verify claims in articles
✔ Social Media Users – Detect misinformation before sharing
✔ Educators & Students – Teach critical thinking with AI-assisted fact-checking
✔ Content Moderators – Flag unreliable news sources

📌 Why AI FactGuard PRO?
✅ Fast & Scalable – Uses Groq's lightning-fast inference
✅ Transparent – Shows reasoning behind each verdict
✅ Customizable – Switch between different AI models
✅ Open & Extensible – Can integrate additional fact-checking APIs
