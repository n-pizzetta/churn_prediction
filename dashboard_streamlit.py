import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Load the saved models
clustering_pipeline = joblib.load('models/clustering_pipeline.joblib')
classification_pipeline = joblib.load('models/classification_pipeline.joblib')


# Load your data
@st.cache_data
def load_data():
    # Load data from CSV or other sources
    df = pd.read_csv('data/data.csv')
    return df

df = load_data()

# Exclude 'id' from features for prediction
X_clustering = df.drop(columns=['id', 'churn'], errors='ignore')

# Predict clusters
df['cluster'] = clustering_pipeline.predict(X_clustering)

X = df.drop(columns=['id', 'churn'], errors='ignore')

# Predict churn probabilities
df['predicted_churn_proba'] = classification_pipeline.predict_proba(X)[:, 1]

# Predict churn classes
df['predicted_churn'] = classification_pipeline.predict(X)


# Map cluster numbers to names (if necessary)
cluster_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
df['cluster_name'] = df['cluster'].map(cluster_names)

# Sidebar for navigation
st.sidebar.title("Customer Churn Dashboard")
selection = st.sidebar.radio(
    "Go to",
    [
        "Cluster Overview",
        "Demographics",
        "Services",
        "Contract & Payment",
        "Churn Analysis",
        "Customer Profiles",
        "Recommendations",
        "Churn Prediction",
    ],
)

# Define functions for each section
def cluster_overview():
    st.header("Cluster Overview")

    # Mean summary
    st.subheader("Cluster Summary (Mean)")
    mean_summary = df.groupby('cluster')[['month_tenure', 'amount_charges_monthly', 'amount_total_charges']].mean().round(2)
    mean_summary.reset_index(inplace=True)
    mean_summary['cluster'] = mean_summary['cluster'].map(cluster_names)
    st.dataframe(mean_summary)

    # Churn rate by cluster using predicted churn
    st.subheader("Predicted Churn Rate by Cluster")
    churn_rate = (
        df.groupby('cluster')['predicted_churn']
        .value_counts(normalize=True)
        .rename('proportion')
        .reset_index()
    )
    churn_rate['cluster'] = churn_rate['cluster'].map(cluster_names)
    churn_rate['predicted_churn'] = churn_rate['predicted_churn'].map({0: 'No', 1: 'Yes'})
    fig = px.bar(
        churn_rate,
        x='cluster',
        y='proportion',
        color='predicted_churn',
        barmode='stack',
        labels={'proportion': 'Proportion'},
    )
    st.plotly_chart(fig)


def churn_prediction():
    st.header("Churn Prediction Analysis")

    # Display overall churn probability distribution
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(
        df,
        x='predicted_churn_proba',
        nbins=50,
        title='Distribution of Predicted Churn Probabilities'
    )
    st.plotly_chart(fig)

    # Churn probability by cluster
    st.subheader("Average Churn Probability by Cluster")
    churn_prob_cluster = df.groupby('cluster')['predicted_churn_proba'].mean().reset_index()
    churn_prob_cluster['cluster'] = churn_prob_cluster['cluster'].map(cluster_names)
    fig = px.bar(
        churn_prob_cluster,
        x='cluster',
        y='predicted_churn_proba',
        labels={'predicted_churn_proba': 'Average Churn Probability'},
    )
    st.plotly_chart(fig)

    # High-risk customers
    st.subheader("High-Risk Customers (Churn Probability > 0.7)")
    high_risk_customers = df[df['predicted_churn_proba'] > 0.7]
    st.write(f"Number of High-Risk Customers: {len(high_risk_customers)}")
    st.dataframe(high_risk_customers[['id', 'cluster_name', 'predicted_churn_proba'] + [col for col in df.columns if 'feature' in col]].head(20))


def demographics():
    st.header("Demographic Distribution")

    # Gender distribution
    st.subheader("Gender Distribution by Cluster")
    gender_dist = df.groupby(['cluster', 'gender']).size().reset_index(name='counts')
    gender_dist['cluster'] = gender_dist['cluster'].map(cluster_names)
    fig = px.bar(
        gender_dist,
        x='cluster',
        y='counts',
        color='gender',
        barmode='group',
    )
    st.plotly_chart(fig)

    # Senior citizen distribution
    st.subheader("Senior Citizen Distribution by Cluster")
    senior_dist = df.groupby(['cluster', 'customer_senior']).size().reset_index(name='counts')
    senior_dist['cluster'] = senior_dist['cluster'].map(cluster_names)
    fig = px.bar(
        senior_dist,
        x='cluster',
        y='counts',
        color='customer_senior',
        barmode='group',
        labels={'customer_senior': 'Senior Citizen'},
    )
    st.plotly_chart(fig)

    # Partner status
    st.subheader("Partner Status by Cluster")
    partner_dist = df.groupby(['cluster', 'customer_partner']).size().reset_index(name='counts')
    partner_dist['cluster'] = partner_dist['cluster'].map(cluster_names)
    fig = px.bar(
        partner_dist,
        x='cluster',
        y='counts',
        color='customer_partner',
        barmode='group',
        labels={'customer_partner': 'Partner'},
    )
    st.plotly_chart(fig)

    # Dependents
    st.subheader("Dependents by Cluster")
    dependents_dist = df.groupby(['cluster', 'dependent_family']).size().reset_index(name='counts')
    dependents_dist['cluster'] = dependents_dist['cluster'].map(cluster_names)
    fig = px.bar(
        dependents_dist,
        x='cluster',
        y='counts',
        color='dependent_family',
        barmode='group',
        labels={'dependent_family': 'Dependents'},
    )
    st.plotly_chart(fig)

def services():
    st.header("Service Subscription Analysis")

    # Internet type
    st.subheader("Internet Service Type by Cluster")
    internet_dist = df.groupby(['cluster', 'internet_type']).size().reset_index(name='counts')
    internet_dist['cluster'] = internet_dist['cluster'].map(cluster_names)
    fig = px.bar(
        internet_dist,
        x='cluster',
        y='counts',
        color='internet_type',
        barmode='group',
    )
    st.plotly_chart(fig)

    # Online security
    st.subheader("Online Security by Cluster")
    security_dist = df.groupby(['cluster', 'online_security']).size().reset_index(name='counts')
    security_dist['cluster'] = security_dist['cluster'].map(cluster_names)
    fig = px.bar(
        security_dist,
        x='cluster',
        y='counts',
        color='online_security',
        barmode='group',
        labels={'online_security': 'Online Security'},
    )
    st.plotly_chart(fig)

    # Streaming TV
    st.subheader("Streaming TV by Cluster")
    tv_dist = df.groupby(['cluster', 'streaming_tv']).size().reset_index(name='counts')
    tv_dist['cluster'] = tv_dist['cluster'].map(cluster_names)
    fig = px.bar(
        tv_dist,
        x='cluster',
        y='counts',
        color='streaming_tv',
        barmode='group',
        labels={'streaming_tv': 'Streaming TV'},
    )
    st.plotly_chart(fig)

def contract_and_payment():
    st.header("Contract and Payment Preferences")

    # Contract type
    st.subheader("Contract Type by Cluster")
    contract_dist = df.groupby(['cluster', 'contract_type']).size().reset_index(name='counts')
    contract_dist['cluster'] = contract_dist['cluster'].map(cluster_names)
    fig = px.bar(
        contract_dist,
        x='cluster',
        y='counts',
        color='contract_type',
        barmode='group',
        labels={'contract_type': 'Contract Type'},
    )
    st.plotly_chart(fig)

    # Payment method
    st.subheader("Payment Method by Cluster")
    payment_dist = df.groupby(['cluster', 'payment_method_type']).size().reset_index(name='counts')
    payment_dist['cluster'] = payment_dist['cluster'].map(cluster_names)
    fig = px.bar(
        payment_dist,
        x='cluster',
        y='counts',
        color='payment_method_type',
        barmode='group',
        labels={'payment_method_type': 'Payment Method'},
    )
    st.plotly_chart(fig)

def churn_analysis():
    st.header("Churn Analysis")

    # Churn rate by cluster using predicted churn
    st.subheader("Predicted Churn Rate by Cluster")
    churn_rate = (
        df.groupby('cluster')['predicted_churn']
        .value_counts(normalize=True)
        .rename('proportion')
        .reset_index()
    )
    churn_rate['cluster'] = churn_rate['cluster'].map(cluster_names)
    churn_rate['predicted_churn'] = churn_rate['predicted_churn'].map({0: 'No', 1: 'Yes'})
    fig = px.bar(
        churn_rate,
        x='cluster',
        y='proportion',
        color='predicted_churn',
        barmode='stack',
        labels={'proportion': 'Proportion'},
    )
    st.plotly_chart(fig)

    # Churn by contract type using predicted churn
    st.subheader("Predicted Churn Rate by Contract Type")
    churn_contract = (
        df.groupby('contract_type')['predicted_churn']
        .value_counts(normalize=True)
        .rename('proportion')
        .reset_index()
    )
    churn_contract['predicted_churn'] = churn_contract['predicted_churn'].map({0: 'No', 1: 'Yes'})
    fig = px.bar(
        churn_contract,
        x='contract_type',
        y='proportion',
        color='predicted_churn',
        barmode='stack',
        labels={'proportion': 'Proportion'},
    )
    st.plotly_chart(fig)

    # Churn by payment method using predicted churn
    st.subheader("Predicted Churn Rate by Payment Method")
    churn_payment = (
        df.groupby('payment_method_type')['predicted_churn']
        .value_counts(normalize=True)
        .rename('proportion')
        .reset_index()
    )
    churn_payment['predicted_churn'] = churn_payment['predicted_churn'].map({0: 'No', 1: 'Yes'})
    fig = px.bar(
        churn_payment,
        x='payment_method_type',
        y='proportion',
        color='predicted_churn',
        barmode='stack',
        labels={'proportion': 'Proportion'},
    )
    st.plotly_chart(fig)

def customer_profiles():
    st.header("Customer Profiles")

    selected_cluster = st.selectbox(
        "Select Cluster",
        options=sorted(df['cluster'].unique()),
        format_func=lambda x: cluster_names.get(x, f"Cluster {x}"),
    )

    cluster_df = df[df['cluster'] == selected_cluster]

    st.subheader(f"{cluster_names.get(selected_cluster, 'Cluster')} ({selected_cluster})")
    st.write(f"Number of Customers: {len(cluster_df)}")

    # Display sample customers
    st.subheader("Sample Customer Data")
    st.dataframe(cluster_df.head(10))

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(cluster_df.describe().round(2))

def recommendations():
    st.header("Marketing Recommendations")

    st.subheader("Cluster 2: Phone Service Only Customers")
    st.markdown(
        """
    - **Upselling Internet Services:** Introduce attractive bundled packages.
    - **Promote Convenience Features:** Highlight benefits of paperless billing.
    - **Loyalty Programs:** Implement rewards for continued patronage.
    """
    )

    st.subheader("Cluster 1: Newer, Price-Sensitive Customers")
    st.markdown(
        """
    - **Retention Efforts:** Implement retention campaigns focusing on satisfaction.
    - **Promote Long-Term Contracts:** Offer incentives for longer-term contracts.
    - **Upsell Add-on Services:** Educate on the value of additional services.
    - **Personalized Communication:** Use targeted messaging.
    """
    )

    st.subheader("Cluster 0: Loyal, High-Value Customers")
    st.markdown(
        """
    - **Enhance Loyalty Programs:** Offer exclusive deals and recognition.
    - **Cross-Selling Opportunities:** Introduce new premium services.
    - **Solicit Feedback:** Engage in feedback programs.
    - **Maintain Service Excellence:** Ensure high-quality customer service.
    """
    )

# Main execution based on selection
if selection == "Cluster Overview":
    cluster_overview()
elif selection == "Demographics":
    demographics()
elif selection == "Services":
    services()
elif selection == "Contract & Payment":
    contract_and_payment()
elif selection == "Churn Analysis":
    churn_analysis()
elif selection == "Customer Profiles":
    customer_profiles()
elif selection == "Recommendations":
    recommendations()
elif selection == "Churn Prediction":
    churn_prediction()
else:
    st.write("Select a section from the sidebar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by TSE students")
