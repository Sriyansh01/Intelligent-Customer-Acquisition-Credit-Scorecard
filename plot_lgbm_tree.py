import streamlit as st
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split

# 1. SETUP
st.set_page_config(page_title="EquiScore Tree Lab", layout="wide")
st.title("🔬 Research Lab: EquiScore Leaf-wise Decision Tree")
st.write("Visualizing the internal logic of the Proposed Framework using **cs-training.csv**.")

# 2. DATA HANDLING (150k Kaggle Features)
@st.cache_data
def load_and_preprocess_150k():
    try:
        df = pd.read_csv("cs-training.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
        X = df.drop('SeriousDlqin2yrs', axis=1)
        y = df['SeriousDlqin2yrs']
        return X, y
    except FileNotFoundError:
        st.error("❌ 'cs-training.csv' not found in the current directory.")
        return None, None

# 3. PLOT GENERATION
def get_tree_graph():
    X, y = load_and_preprocess_150k()
    if X is not None:
        lgb_train = lgb.Dataset(X, y, feature_name=list(X.columns), free_raw_data=False)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_leaves': 12,
            'learning_rate': 0.1,
            'verbose': -1
        }
        lgb_model = lgb.train(params, lgb_train, num_boost_round=1)
        plot = lgb.create_tree_digraph(
            lgb_model, 
            tree_index=0, 
            show_info=['split_gain'],
            node_attr={'shape': 'box', 'style': 'filled', 'color': 'lightblue'},
            edge_attr={'color': 'darkgrey'},
            graph_attr={'rankdir': 'TB'} 
        )
        return plot
    return None

# 4. RENDERING
tree_output = get_tree_graph()

if tree_output is not None:
    dot_string = str(tree_output.source)
    st.graphviz_chart(dot_string, use_container_width=True)
    st.success("✅ Tree successfully rendered.")
    st.info("Analysis: This vertical structure highlights the Leaf-wise growth strategy. It prioritizes nodes with the highest information gain, capturing deep patterns in credit behavior.")