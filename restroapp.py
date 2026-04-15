import streamlit as st
import altair as alt
import pandas as pd 
from numpy.random import default_rng as rng, random
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations",
                    page_icon="🍷🍽️", layout="wide")
st.title("👩🏻‍🍳🥘🤌🏻Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations🍴😋🍕")

@st.cache_data
def load_data():
    data=pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")
    return data

df=load_data()
df = df.fillna(0)

#Freature Engineering
df["Total_Revenue"] = (df["InStoreRevenue"] + df["UberEatsRevenue"] + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"])
df["TotalOrders"] = (df["InStoreOrders"] + df["UberEatsOrders"] + df["DoorDashOrders"] + df["SelfDeliveryOrders"])

#Channel Revenue Ratio
df["InStoreRevenueRatio"] = df["InStoreRevenue"] / df["Total_Revenue"]
df["UberEatsRevenueRatio"] = df["UberEatsRevenue"] / df["Total_Revenue"]
df["DoorDashRevenueRatio"] = df["DoorDashRevenue"] / df["Total_Revenue"]
df["SelfDeliveryRevenueRatio"] = df["SelfDeliveryRevenue"] / df["Total_Revenue"]

#Cost to Revenue Ratio
df["TotalCostRate"] = df["COGSRate"] + df["OPEXRate"]
df["CostRevenue"] = df["TotalCostRate"] / df["Total_Revenue"]

#Profit Margin
df["TotalNetProfit"] = (df["InStoreRevenue"] + df["UberEatsRevenue"] + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"])
df["ProfitPerOrder"] = df["TotalNetProfit"] / df["TotalOrders"]

#Interactive Features
df["Commission_UE_Impact"] = df["CommissionRate"] * df["UberEatsRevenue"]
df["Commission_SD_Impact"] = df["CommissionRate"] * df["SelfDeliveryRevenue"]

#Growth Adjustment Demand
df["Adj_TotalOrders"] = df["TotalOrders"] * (1 + df["GrowthFactor"])
df["Adj_TotalRevenue"] = df["Total_Revenue"] * (1 + df["GrowthFactor"])

df = pd.get_dummies(df,columns=["CuisineType", "Segment", "Subregion"], drop_first=True)

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
for col in ["Total_Revenue", "TotalOrders", "ProfitPerOrder"]:
 df = remove_outliers(df, col)

st.sidebar.header("Filter Restaurants")
numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
target = st.sidebar.selectbox("Select Target(Profit)", numeric_cols)
feature_options = [col for col in numeric_cols if col != target]
default_features = feature_options[:5]
features = st.sidebar.multiselect("Select Features", feature_options, default=default_features)




tab1, tab2, tab3, tab4 = st.tabs(["🎯Performance Dashboard","📊Model Development","💰Profit Prediction","🔎What-If Analysis", ])
with tab1:
     
    total_revenue = df[[
        "InStoreRevenue",
        "UberEatsRevenue",
        "DoorDashRevenue",
        "SelfDeliveryRevenue"
    ]].sum().sum()

    total_orders = df[[
        "InStoreOrders",
        "UberEatsOrders",
        "DoorDashOrders",
        "SelfDeliveryOrders"
    ]].sum().sum()

    avg_order_value = total_revenue / total_orders

    cogs = total_revenue * df["COGSRate"].mean()
    opex = total_revenue * df["OPEXRate"].mean()

    predicted_profit = total_revenue - (cogs + opex)

    profit_margin = (predicted_profit / total_revenue) * 100

    channel_efficiency = (df["InStoreRevenue"].sum() / total_revenue) * 100

    break_even_commission = df["CommissionRate"].mean() * 100

    optimization_uplift = profit_margin * 0.15  # simulated


    st.markdown("""
    <style>
    .kpi-card {
        padding: 20px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        transition: 0.3s;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    .kpi-title {
        font-size: 14px;
        opacity: 0.9;
   }
   .kpi-value {
        font-size: 28px;
        font-weight: bold;
    }
    .icon {
        font-size: 30px;
    }
    </style>
    """, unsafe_allow_html=True)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg,#667eea,#764ba2);">
            <div class="icon">💰</div>
            <div class="kpi-title">Predicted Net Profit</div>
            <div class="kpi-value">₹{predicted_profit:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#43cea2,#185a9d);">
            <div class="icon">📈</div>
            <div class="kpi-title">Profit Margin</div>
            <div class="kpi-value">{profit_margin:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#f7971e,#ffd200);">
            <div class="icon">⚡</div>
            <div class="kpi-title">Channel Efficiency</div>
            <div class="kpi-value">{channel_efficiency:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#ff6a00,#ee0979);">
            <div class="icon">🎯</div>
            <div class="kpi-title">Break-Even Commission</div>
            <div class="kpi-value">{break_even_commission:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# 💎 KPI CARDS (ROW 2)
# =========================
    col5, col6, col7 = st.columns(3)

    with col5:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#11998e,#38ef7d);">
            <div class="icon">📦</div>
            <div class="kpi-title">Total Orders</div>
            <div class="kpi-value">{int(total_orders):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#fc466b,#3f5efb);">
            <div class="icon">🧾</div>
            <div class="kpi-title">Avg Order Value</div>
            <div class="kpi-value">₹{avg_order_value:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg,#8360c3,#2ebf91);">
            <div class="icon">🚀</div>
            <div class="kpi-title">Optimization Uplift</div>
            <div class="kpi-value">{optimization_uplift:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

                                                                             
    fig3 = px.scatter(
        df,
        x="AOV",
        y="InStoreNetProfit",
        labels={
            "AOV": "Average Order Value",
            "InStoreNetProfit": "In-Store Net Profit",
        },
        title="💲Average order value vs Profit", color_discrete_sequence=["#D0DE0F"]
    )
    st.plotly_chart(fig3, use_container_width=True, height=400)
 

with tab2:
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }
    results = {}
    for name, model in models.items():
       model.fit(X_train, y_train)
       pred = model.predict(X_test)
       rmse = np.sqrt(np.mean((y_test - pred) ** 2))
       r2 = r2_score(y_test, pred)
       mae = mean_absolute_error(y_test, pred)
       results[name] = {"RMSE": rmse, "R2 Score": r2, "MAE": mae}
       results_df = pd.DataFrame.from_dict(results, orient="index")
    st.subheader("🚀Model Performance")
    st.dataframe(results_df)
    best_model_name = results_df.sort_values("R2 Score",ascending=False).iloc[0].name
    st.write(f"Best Model: {best_model_name}")
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)



    st.subheader("📶Target Distribution")
    fig4 = px.histogram(df, x=target, nbins=30, title=f"Distribution of {target}", color_discrete_sequence=["#A54CAF"])
    st.plotly_chart(fig4, use_container_width=True)


    st.header("✍🏻Scenario Simulation")
    st.markdown("Use the sliders below to simulate changes in key features and see the predicted profit impact.")
    input_data = {}
    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        input_data[feature] = cols[i].slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_df = pd.DataFrame([input_data])
    predicted_profit = best_model.predict(input_df)[0]
    st.metric("Predicted Profit", f"{predicted_profit:2f}")


    col1, col2 = st.columns(2)
    with col1:
       if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False) 
        fig5 = px.line(importance, x=importance.index, y=importance.values,
                        title="Feature Importance", color_discrete_sequence=["#AF4F4C"])
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
       
        if hasattr(best_model, "feature_importances_"):
            importance = pd.Series(
               best_model.feature_importances_,
               index=features
            ).sort_values(ascending=False)

             # Optional: Top 5 + Others (recommended for clean look)
            top_n = 5
            top_features = importance[:top_n]
            others = importance[top_n:].sum()

            importance_clean = pd.concat([
            top_features,
            pd.Series({"Others": others})
            ])

            # Custom distinct colors (Power BI style)
            colors = ["#EF4FCA", "#6A2BF2", "#9A1316", "#76B7B2", "#59A14F", "#EDC948"]

            fig5 = px.pie(
            names=importance_clean.index,
            values=importance_clean.values,
            hole=0.6,
            title="Feature Importance Share",
            color=importance_clean.index,
            color_discrete_sequence=colors
            )

            fig5.update_traces(
            textinfo="percent+label",
            pull=[0.06 if i == 0 else 0 for i in range(len(importance_clean))]
            )

            fig5.update_layout(
            title_x=0.5,
            showlegend=True
            )

            st.plotly_chart(fig5, use_container_width=True)


with tab3:
    
    change_percent = st.slider("Adjust feature(%)", -50, 50, 0)
    modified_input = input_df.copy()
    modified_input = modified_input * (1 + change_percent / 100)
    new_prediction = best_model.predict(modified_input)[0]
    col1,col2 = st.columns(2)
    col1.metric("Original Predicted Profit", f"{predicted_profit:2f}")
    col2.metric("Modified Predicted Profit", f"{new_prediction:2f}")

    st.header("📃Prescriptive Optimization")
    st.markdown("""
            -Identify profit-maximizing inputs
            -Recommend optimal feature values""")
    best_profit = -np.inf
    best_config = None
    for _ in range(100):
       random_sample = {col: np.random.uniform(df[col].min(), df[col].max()) for col in features}
       sample_df = pd.DataFrame([random_sample])
       pred = best_model.predict(sample_df)[0]
    if pred > best_profit:
       best_profit = pred
       best_config = random_sample


    st.subheader("🌀Optimal Feature Configuration")
    st.metric("Max Predicted Profit", f"{best_profit:2f}")

    st.subheader("💸Profit Contribution by Channel")
    revenue_cols = ["InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue"]
    values = [df[col].sum() for col in revenue_cols]
    wf = pd.DataFrame({
    "Channel": revenue_cols,
    "Value": values
    })
    wf["Cumulative"] = wf["Value"].cumsum()
    wf["Start"] = wf["Cumulative"] - wf["Value"]
    wf["End"] = wf["Cumulative"]
    total = wf["Value"].sum()
    wf_total = pd.DataFrame({
    "Channel": ["Total"],
    "Value": [total],
    "Start": [0],
    "End": [total]
    })

    wf = pd.concat([wf, wf_total], ignore_index=True)

    # Step 4: Create chart
    chart = alt.Chart(wf).mark_bar().encode(
    x=alt.X("Channel:N", title="Channel"),
    y=alt.Y("Start:Q", title="Profit"),
    y2="End:Q",
    color=alt.condition(
        alt.datum.Channel == "Total",
        alt.value("lightpink"),
        alt.value("#4CAF50")
    ),
    tooltip=["Channel", "Value"]
    )

    st.altair_chart(chart, use_container_width=True)

   

with tab4:
    
   # Clean column names (IMPORTANT)
    df.columns = df.columns.str.strip()

    st.sidebar.header("🔍 Filter Dashboard")

    min_rev, max_rev = st.sidebar.slider(
    "Total Revenue Range",
    float(df["InStoreRevenue"].min()),
    float(df["InStoreRevenue"].max()),
    (
        float(df["InStoreRevenue"].min()),
        float(df["InStoreRevenue"].max())
    )
    )

    min_orders, max_orders = st.sidebar.slider(
    "Monthly Orders",
    int(df["MonthlyOrders"].min()),
    int(df["MonthlyOrders"].max()),
    (
        int(df["MonthlyOrders"].min()),
        int(df["MonthlyOrders"].max())
    )
    )

    min_growth, max_growth = st.sidebar.slider(
    "Growth Factor",
    float(df["GrowthFactor"].min()),
    float(df["GrowthFactor"].max()),
    (
        float(df["GrowthFactor"].min()),
        float(df["GrowthFactor"].max())
    )
    )

# -------------------------------
# ✅ 5. Channel Selection
# -------------------------------
    channels = st.sidebar.multiselect(
    "Select Sales Channels",
    ["InStore", "UberEats", "DoorDash", "SelfDelivery"],
    default=["InStore", "UberEats", "DoorDash", "SelfDelivery"]
    )

# -------------------------------
# ✅ 6. Cost Filters
# -------------------------------
    min_cogs, max_cogs = st.sidebar.slider(
    "COGS Rate",
    float(df["COGSRate"].min()),
    float(df["COGSRate"].max()),
    (
        float(df["COGSRate"].min()),
        float(df["COGSRate"].max())
    )
    )

    min_opex, max_opex = st.sidebar.slider(
    "OPEX Rate",
    float(df["OPEXRate"].min()),
    float(df["OPEXRate"].max()),
    (
        float(df["OPEXRate"].min()),
        float(df["OPEXRate"].max())
    )
    )

# -------------------------------
# ✅ APPLY FILTERS
# -------------------------------
    filtered_df = df[
    (df["InStoreRevenue"].between(min_rev, max_rev)) &
    (df["MonthlyOrders"].between(min_orders, max_orders)) &
    (df["GrowthFactor"].between(min_growth, max_growth)) &
    (df["COGSRate"].between(min_cogs, max_cogs)) &
    (df["OPEXRate"].between(min_opex, max_opex))
    ]

# -------------------------------
# ✅ CHANNEL-BASED REVENUE CALC
# -------------------------------
    channel_columns = []

    if "InStore" in channels:
        channel_columns.append("InStoreRevenue")
    if "UberEats" in channels:
        channel_columns.append("UberEatsRevenue")
    if "DoorDash" in channels:
        channel_columns.append("DoorDashRevenue")
    if "SelfDelivery" in channels:
        channel_columns.append("SelfDeliveryRevenue")

    filtered_df["SelectedChannelRevenue"] = filtered_df[channel_columns].sum(axis=1)




    st.subheader("🔑 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue", f"{filtered_df['SelectedChannelRevenue'].sum():,.0f}")
    col2.metric("Total Orders", f"{filtered_df['MonthlyOrders'].sum():,.0f}")
    col3.metric("Avg Growth", f"{filtered_df['GrowthFactor'].mean():.2f}")
# -------------------------------
# ✅ SHOW DATA
# -------------------------------
    st.subheader("📊 Filtered Data")
    st.dataframe(filtered_df)

    
    import plotly.express as px

    revenue_cols = ["InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue"]
    df_melted = df.melt(
    value_vars=revenue_cols,
    var_name="Channel",
    value_name="Revenue"
     )
    fig = px.bar( df_melted, x="Revenue", y="Channel",
    color="Channel",
    orientation="h"
    )

    fig.update_layout(
    barmode="stack",   
    title="💰Revenue Distribution by Channel",
    xaxis_title="Revenue",
    yaxis_title="Channel"
    )
    st.plotly_chart(fig, use_container_width=True)

   
    st.sidebar.header("⚙️ What-If Controls")

    instore_share = st.sidebar.slider("In-Store %", 0, 100, 40)
    ubereats_share = st.sidebar.slider("UberEats %", 0, 100, 25)
    doordash_share = st.sidebar.slider("DoorDash %", 0, 100, 20)
    self_delivery_share = st.sidebar.slider("Self Delivery %", 0, 100, 15)

    commission_rate = st.sidebar.slider("Commission Rate (%)", 5, 40, 20)
    delivery_cost = st.sidebar.slider("Delivery Cost per Order (₹)", 10, 150, 50)
    cogs_rate = st.sidebar.slider("COGS (%)", 10, 60, 35)
    opex_rate = st.sidebar.slider("OPEX (%)", 5, 40, 20)


    total_share = instore_share + ubereats_share + doordash_share + self_delivery_share
    instore_share /= total_share
    ubereats_share /= total_share
    doordash_share /= total_share
    self_delivery_share /= total_share

    revenue_cols = [
    "InStoreRevenue",
    "UberEatsRevenue",
    "DoorDashRevenue",
    "SelfDeliveryRevenue"
    ]

    df["TotalRevenue"] = df[revenue_cols].sum(axis=1)
    base_revenue = df["TotalRevenue"].sum()
    total_orders = df["TotalOrders"].sum()


    instore_rev = base_revenue * instore_share
    ubereats_rev = base_revenue * ubereats_share
    doordash_rev = base_revenue * doordash_share
    self_rev = base_revenue * self_delivery_share

  # Costs
    commission_cost = (ubereats_rev + doordash_rev) * (commission_rate / 100)
    delivery_total = total_orders * delivery_cost
    cogs = base_revenue * (cogs_rate / 100)
    opex = base_revenue * (opex_rate / 100)


    predicted_profit = base_revenue - (commission_cost + delivery_total + cogs + opex)

   # Confidence band (simulated uncertainty)
    uncertainty = predicted_profit * 0.1
    lower_bound = predicted_profit - uncertainty
    upper_bound = predicted_profit + uncertainty

   # Risk indicator
    risk_score = (commission_rate * 0.4 + delivery_cost * 0.3 + cogs_rate * 0.3)

    
    st.subheader("📊 Revenue Breakdown by Channel")
    channel_df = pd.DataFrame({
           "Channel": ["InStore", "UberEats", "DoorDash", "Self Delivery"],
           "Revenue": [instore_rev, ubereats_rev, doordash_rev, self_rev]
        })

    fig = px.pie(channel_df, names="Channel", values="Revenue", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("📉 Cost Sensitivity Analysis")

    commission_range = np.linspace(5, 40, 20)
    profit_sensitivity = []

    for c in commission_range:
       cost = (ubereats_rev + doordash_rev) * (c / 100)
       profit = base_revenue - (cost + delivery_total + cogs + opex)
       profit_sensitivity.append(profit)

    sens_df = pd.DataFrame({
       "Commission %": commission_range,
       "Profit": profit_sensitivity
   })

    fig2 = px.line(sens_df, x="Commission %", y="Profit", title="Profit vs Commission")
    st.plotly_chart(fig2, use_container_width=True)


    st.subheader("🧠 Optimization Recommendations")

    recommendations = []

    if commission_rate > 25:
       recommendations.append("🔻 Reduce reliance on aggregator platforms (high commission).")

    if delivery_cost > 80:
       recommendations.append("🚚 Optimize delivery logistics or introduce delivery fees.")

    if instore_share < 0.3:
       recommendations.append("🏪 Boost in-store promotions to improve margins.")

    if cogs_rate > 40:
       recommendations.append("📦 Negotiate supplier costs or adjust menu pricing.")

    if not recommendations:
       recommendations.append("✅ Current strategy looks optimal.")

    for rec in recommendations:
       st.write(rec)

   # =========================
   # 🚨 RISK INDICATOR
   # =========================
    st.subheader("🚨 Risk Indicator")

    if risk_score < 30:
       st.success("Low Risk 🟢")
    elif risk_score < 60:
       st.warning("Moderate Risk 🟡")
    else:
       st.error("High Risk 🔴")
