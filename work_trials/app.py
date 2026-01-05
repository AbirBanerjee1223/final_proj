import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv() 

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


st.set_page_config(
    page_title="Essential Protein Identification",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS - Updated with Linear Gradient Background
st.markdown("""
    <style>
    /* 1. Main Background - GRADIENT OVERRIDE */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #E9EDF3 0%, #DDE3EA 100%) !important;
        background-attachment: fixed !important; /* Keeps gradient fixed while scrolling */
    }

    /* 2. Sidebar Background */
    section[data-testid="stSidebar"], [data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
    }

    /* 3. Cards / Containers (White Backgrounds) */
    .feature-box { 
        background-color: #FFFFFF !important; 
        padding: 1rem; 
        border-radius: 5px; 
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
        color: #31333F; 
    }

    .result-box { 
        padding: 2rem; 
        border-radius: 10px; 
        text-align: center; 
        font-size: 1.5rem; 
        font-weight: bold; 
        margin: 2rem 0; 
        background-color: #FFFFFF !important; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #31333F;
    }
    
    /* Specific overrides for result states */
    .essential { 
        background-color: #d4edda !important; 
        color: #155724 !important; 
        border: 2px solid #c3e6cb !important; 
    }
    .non-essential { 
        background-color: #f8d7da !important; 
        color: #721c24 !important; 
        border: 2px solid #f5c6cb !important; 
    }

    /* AI Insight Box Styling */
    .ai-box {
        background-color: #F0F9FF !important;
        border: 1px solid #BAE6FD !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        color: #0C4A6E;
        font-size: 0.95rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Header Styling */
    .main-header { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #1f77b4; 
        text-align: center; 
        margin-bottom: 0.5rem; 
    }
    .sub-header { 
        font-size: 1.2rem; 
        color: #666; 
        text-align: center; 
        margin-bottom: 2rem; 
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dropdown_data():
    """Loads the list of proteins for the dropdown with safety cleaning"""
    path = 'D:\\python_progs\\final_proj\\notebooks\\protein_dropdown.csv'
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = df.dropna(subset=['Display_Name', 'Entry'])
            df['Display_Name'] = df['Display_Name'].astype(str).str.strip()
            df['Entry'] = df['Entry'].astype(str).str.strip()
            return df
        except Exception as e:
            st.error(f"Error reading protein list: {e}")
            return pd.DataFrame(columns=['Display_Name', 'Entry'])
    return pd.DataFrame(columns=['Display_Name', 'Entry'])

@st.cache_data
def load_feature_database():
    """Loads the full dataset to fetch features for prediction"""
    path = 'D:\\python_progs\\final_proj\\notebooks\\Full_database.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_resources():
    """Loads trained models and scaler"""
    resources = {}
    try:
        # Load Scaler
        if os.path.exists('scaler.pkl'):
            resources['scaler'] = joblib.load('scaler.pkl')
        
        # Load XGBoost 
        if os.path.exists('xgb_essentiality_final.pkl'):
            resources['XGBoost'] = joblib.load('xgb_essentiality_final.pkl')
        
        # Load Random Forest 
        if os.path.exists('rf_essentiality_final.pkl'):
            resources['Random Forest'] = joblib.load('rf_essentiality_final.pkl')
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return resources

# Load data on app startup
dropdown_df = load_dropdown_data()
feature_db = load_feature_database()
resources = load_resources()


# Mapping actual column names to readable descriptions
FEATURE_DESCRIPTIONS = {
    "PCP_Hydrophobicity": "Average hydrophobicity of the sequence. Essential cores are often hydrophobic.",
    "PCP_Charge": "Net charge of the protein. Affects interactions with DNA/RNA.",
    "SER_Entropy": "Shannon Entropy measures sequence complexity. Low complexity often indicates structural roles.",
    "Aromaticity": "Proportion of aromatic amino acids (Phe, Trp, Tyr). Important for stability.",
    "Protein_Length": "Total number of amino acids. Essential scaffolds tend to be larger.",
    "Degree_Centrality": "Number of direct partners in the PPI network. High degree = Hub protein.",
    "Betweenness_Centrality": "How often the protein acts as a bridge in the network.",
    "Closeness_Centrality": "How quickly this protein can reach others in the network.",
    "Eigenvector_Centrality": "Influence of the protein based on connection to other important proteins.",
    "Loc_Nucleus": "Localized in the Nucleus. Strong indicator of essentiality (Replication/Transcription).",
    "Loc_Membrane": "Localized in the Membrane. Often receptors/transporters (Lower essentiality risk).",
    "Loc_Cytoplasm": "Localized in the Cytoplasm. Metabolic enzymes often reside here.",
    "Loc_Mitochondria": "Localized in Mitochondria. Critical for energy production."
}

def create_feature_importance_plot(features_dict):
    """Horizontal bar chart of scaled feature values"""
    # Sort by absolute value to show most impactful features
    sorted_feats = sorted(features_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    keys = [x[0] for x in sorted_feats]
    values = [x[1] for x in sorted_feats]
    
    colors = ['#ff7f0e' if v > 0 else '#1f77b4' for v in values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=keys,
        x=values,
        orientation='h',
        marker=dict(color=colors)
    ))
    
    fig.update_layout(
        title="Top Contributing Features (Scaled)",
        xaxis_title="Normalized Value (Impact)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool identifies essential proteins in the human PPI network using:
    
    - Uses topological features (Centrality, Degree)
    - Integrates Physicochemical properties
    - Considers Subcellular Localization
    """)
    
    st.markdown("---")
    st.markdown("**📊 Features**")
    st.success("""
    ⏱️Real-time Prediction\n
    📈Feature Importance Visualization\n
    🔗Interactive Tooltips\n
    📊Model Performance Metrics\n
    🤖Gemini AI Explanations
    """)
    
    st.markdown("---")
    st.markdown("**Model Information**")
    st.info("Predictions are generated using pre-trained **XGBoost** and **Random Forest** models.")
    
    st.markdown("---")
    st.markdown("**Sample Inputs**")
    with st.expander("Example Protein IDs"):
        st.code("P04637 (TP53)\nQ7Z7A1 (Centriolin CEP110)\nO60225 (Protein SSX5)")


st.markdown('<div class="main-header">🧬 Essential Protein Identification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze proteins using machine learning & network topology</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analysis Dashboard", "📊 Model Performance"])

with tab1:
    st.markdown("---")
    

    # INPUT SECTION

    col_input, col_result = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("📋 Input")
        
        # Model Selection
        available_models = [k for k in resources.keys() if k != 'scaler']
        if not available_models:
            st.error("No models found in 'Models/' folder.")
            model_choice = None
        else:
            model_choice = st.selectbox("Select Model:", available_models, index=0)

        st.markdown("---")
        
        # Protein Selection 
        selected_entry_id = None
        selected_display_name = "Unknown Protein"
        
        if not dropdown_df.empty:
            selected_name = st.selectbox(
                "Search Protein Name:", 
                options=dropdown_df['Display_Name'].unique(),
                index=None,  
                placeholder="Type to search (e.g., TP53)...", 
                help="Start typing to filter the protein list"
            )
            
            # Retrieve ID safely
            if selected_name: # Only check if user selected something
                try:
                    row_match = dropdown_df[dropdown_df['Display_Name'] == selected_name]
                    if not row_match.empty:
                        selected_entry_id = row_match['Entry'].values[0]
                        selected_display_name = selected_name
                        st.info(f"**UniProt ID:** {selected_entry_id}")
                    else:
                        st.error("Error retrieving ID for selected protein.")
                except Exception as e:
                    st.error(f"Selection Error: {e}")
            
        else:
            st.warning("Dropdown data not loaded. Check 'Datasets/protein_dropdown.csv'.")

        # Analyze Button
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Predict Essentiality", type="primary", use_container_width=True)


    # RESULT SECTION

    with col_result:
        st.subheader("📊 Analysis Results")
        
        if analyze_btn and selected_entry_id and model_choice:
            with st.spinner(f"Analyzing {selected_entry_id} using {model_choice}..."):
                
                # Fetch Data
                row = feature_db[feature_db['Protein'] == selected_entry_id]
                
                if not row.empty:
                    # Prepare Features
                    drop_cols = ['Protein', 'Gene Names', 'Sequence', 'Raw_Location', 'Label']
                    X_raw = row.drop(columns=[c for c in drop_cols if c in row.columns], errors='ignore')
                    
                    # Preprocessing
                    if 'scaler' in resources:
                        try:
                            scaler = resources['scaler']
                            X_scaled = scaler.transform(X_raw)
                            model = resources[model_choice]
                            prob = model.predict_proba(X_scaled)[:, 1][0]
                            
                            threshold = 0.45
                            is_essential = prob >= threshold
                            label_text = "ESSENTIAL" if is_essential else "NON-ESSENTIAL"
                            
                            # D. Display Prediction
                            if is_essential:
                                st.markdown(f"""
                                    <div class="result-box essential">
                                        ✅ ESSENTIAL PROTEIN<br>
                                        <span style="font-size: 1rem;">Confidence: {prob:.2%}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class="result-box non-essential">
                                        ❌ NON-ESSENTIAL PROTEIN<br>
                                        <span style="font-size: 1rem;">Confidence: {prob:.2%}</span>
                                    </div>
                                """, unsafe_allow_html=True)


                            if api_key:
                                with col_input:
                                    st.markdown("---")
                                    st.subheader("🤖 Gemini AI Insight")
                                    with st.spinner("Generating biological explanation..."):
                                        try:
                                            prompt = (
                                                f"I am analyzing the protein {selected_display_name} (UniProt ID: {selected_entry_id}). "
                                                f"My machine learning model predicted it is **{label_text}** for human or cancer cell, "
                                                f"with {prob:.1%} confidence.\n\n"
                                                f"1. Briefly explain the biological function of this protein.\n"
                                                f"2. Explain why this function might make it {label_text.lower()} for a human or cancer cell. "
                                                f"Keep the response concise (under 150 words) and professional."
                                            )
                                            
                                            # Call Gemini
                                            model_ai = genai.GenerativeModel('gemini-2.5-flash')
                                            response = model_ai.generate_content(prompt)
                                            
                                            # Display Response
                                            st.markdown(f"""
                                                <div class="ai-box">
                                                    {response.text}
                                                </div>
                                            """, unsafe_allow_html=True)
                                            
                                        except Exception as e:
                                            st.error(f"AI Error: {str(e)}")
                            else:
                                with col_input:
                                    st.warning("⚠️ Add GEMINI_API_KEY to .env to see AI explanations.")

                            st.markdown("### 🔬 Feature Analysis")
                            
                            features_dict = dict(zip(X_raw.columns, X_scaled[0]))
                            
                            # Show Top Features Explanation
                            sorted_impact = sorted(features_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                            
                            for feat_name, scaled_val in sorted_impact:
                                raw_val = X_raw.iloc[0][feat_name]
                                desc = FEATURE_DESCRIPTIONS.get(feat_name, "Topological or biophysical property.")
                                
                                with st.expander(f"📌 {feat_name}: {raw_val:.4f}"):
                                    st.write(desc)
                                    norm_val = min(max((scaled_val + 3) / 6, 0.0), 1.0)
                                    st.progress(norm_val)

                            st.markdown("---")
                            st.plotly_chart(create_feature_importance_plot(features_dict), use_container_width=True)
                            st.info("Orange bars indicate positive values (higher than average), Blue bars indicate negative values (lower than average).")
                            
                        except Exception as e:
                            st.error(f"Preprocessing Error: {e}. Ensure feature columns match the training data.")
                    else:
                        st.error("Scaler not found. Cannot process features.")
                else:
                    st.error(f"Protein **{selected_entry_id}** not found in the localized feature database.")
        
        elif analyze_btn and not selected_entry_id:
            st.warning("Please select a protein first.")

# TAB 2: MODEL METRICS

with tab2:
    st.header("📈 Model Validation Metrics")
    st.markdown("Performance evaluated on an independent test set (20% split).")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "78%")
    col2.metric("F1-Score", "0.77")
    col3.metric("Recall (Sensitivity)", "0.78")
    col4.metric("Precision", "0.77")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix Analysis")
        st.info("""
        **Why is Recall higher?**
        We optimized the model threshold (0.45) to prioritize **Recall**. 
        In cancer drug discovery, missing a true target (False Negative) is much worse than flagging a non-essential protein (False Positive).
        """)
    
    with c2:
        st.subheader("Model Architecture")
        st.markdown("""
        - **Algorithm:** XGBoost / Random Forest
        - **Features:** Graph Topology + Physicochemical + Localization
        - **Optimization:** Grid Search with Class Balancing
        - **Validation:** Stratified K-Fold Cross Validation
        """)