# Employee Performance Dashboard
# Author: Tobechukwu Edwin © 2025
# Interactive Dashboard built with Dash & Plotly

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# LOAD & CLEAN DATA
df = pd.read_csv("cleaned_employee_performance.csv")

# Parse hire dates
if "Hire_Date" in df.columns:
    df["Hire_Date"] = pd.to_datetime(df["Hire_Date"], errors="coerce")

# Convert numeric columns safely
for col in ["Performance_Score", "Employee_Satisfaction_Score", "Monthly_Salary",
            "Projects_Handled", "Overtime_Hours", "Productivity_Score",
            "Work_Hours_Per_Week", "Training_Hours", "Tenure_Years"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# KPIs
total_employees = df.shape[0]
resigned_rate = (df["Resigned"].mean()) * 100
avg_perf = df["Performance_Score"].mean()
avg_satisfaction = df["Employee_Satisfaction_Score"].mean()
avg_productivity = df["Productivity_Score"].mean()


# MODEL: Logistic Regression
features = ["Performance_Score", "Employee_Satisfaction_Score", "Productivity_Score",
            "Overtime_Hours", "Training_Hours", "Tenure_Years", "Monthly_Salary"]
X = df[features].fillna(0)
y = df["Resigned"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": abs(model.coef_[0])
}).sort_values("Importance", ascending=False)


# DASH APP SETUP
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("💼 Employee Performance Dashboard", 
            style={"textAlign": "center", "color": "#007bff", "marginBottom": "10px"}),

    html.P("Analyze workforce performance, satisfaction, productivity, and resignation patterns interactively.",
           style={"textAlign": "center", "color": "#555", "marginBottom": "30px"}),

    # KPI Cards Row
    html.Div([
        html.Div([
            html.H4("👥 Total Employees", style={"color": "#555", "fontWeight": "600"}),
            html.H3(f"{total_employees:,}", style={"color": "#007bff", "margin": "0"})
        ], className="card"),
        html.Div([
            html.H4("📉 Resignation Rate", style={"color": "#555", "fontWeight": "600"}),
            html.H3(f"{resigned_rate:.1f}%", style={"color": "#dc3545", "margin": "0"})
        ], className="card"),
        html.Div([
            html.H4("⭐ Avg Performance", style={"color": "#555", "fontWeight": "600"}),
            html.H3(f"{avg_perf:.2f}", style={"color": "#28a745", "margin": "0"})
        ], className="card"),
        html.Div([
            html.H4("😊 Avg Satisfaction", style={"color": "#555", "fontWeight": "600"}),
            html.H3(f"{avg_satisfaction:.2f}", style={"color": "#17a2b8", "margin": "0"})
        ], className="card"),
        html.Div([
            html.H4("⚙ Avg Productivity", style={"color": "#555", "fontWeight": "600"}),
            html.H3(f"{avg_productivity:.2f}", style={"color": "#ffc107", "margin": "0"})
        ], className="card"),
    ], style={
        "display": "flex", "justifyContent": "space-between", "margin": "30px 0",
        "flexWrap": "wrap", "gap": "10px"
    }),

    # Sidebar Filters
    html.Div([
        html.Label("🏢 Department", style={"fontWeight": "600"}),
        dcc.Dropdown(
            id="dept-filter",
            options=[{"label": d, "value": d} for d in sorted(df["Department"].dropna().unique())],
            multi=True, placeholder="Select department(s)"
        ),
        html.Br(),

        html.Label("⚧ Gender", style={"fontWeight": "600"}),
        dcc.Dropdown(
            id="gender-filter",
            options=[{"label": g, "value": g} for g in sorted(df["Gender"].dropna().unique())],
            multi=True, placeholder="Select gender(s)"
        ),
        html.Br(),
        html.P("🔍 Use filters to explore workforce insights", 
               style={"color": "#777", "fontSize": "13px", "fontStyle": "italic"}),
    ], style={
        "width": "22%", "display": "inline-block", "verticalAlign": "top",
        "background": "#f8f9fa", "padding": "20px", "borderRadius": "10px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.1)", "height": "100%"
    }),

    # Main Dashboard Area
    html.Div([
        dcc.Graph(id="perf-vs-satisfaction"),
        dcc.Graph(id="prod-by-dept"),
        dcc.Graph(id="resignation-by-dept"),
        dcc.Graph(id="tenure-vs-resign"),
        dcc.Graph(id="corr-matrix"),
        dcc.Graph(id="confusion-matrix"),
        dcc.Graph(id="feature-importance"),
    ], style={
        "width": "75%", "display": "inline-block", "paddingLeft": "25px",
        "verticalAlign": "top"
    }),

    html.Hr(),
    html.P("© 2025 Tobechukwu Edwin | Interactive HR Analytics Dashboard",
           style={"textAlign": "center", "color": "#777", "fontSize": "13px", "marginTop": "10px"})
], style={
    "fontFamily": "Segoe UI, sans-serif", 
    "padding": "20px", 
    "background": "#f6f7fb"
})


# CALLBACKS
@app.callback(
    [Output("perf-vs-satisfaction", "figure"),
     Output("prod-by-dept", "figure"),
     Output("resignation-by-dept", "figure"),
     Output("tenure-vs-resign", "figure"),
     Output("corr-matrix", "figure"),
     Output("confusion-matrix", "figure"),
     Output("feature-importance", "figure")],
    [Input("dept-filter", "value"),
     Input("gender-filter", "value")]
)
def update_charts(dept_filter, gender_filter):
    dff = df.copy()

    if dept_filter:
        dff = dff[dff["Department"].isin(dept_filter)]
    if gender_filter:
        dff = dff[dff["Gender"].isin(gender_filter)]

    # Performance vs Satisfaction
    fig1 = px.scatter(dff, x="Employee_Satisfaction_Score", y="Performance_Score",
                      color="Resigned", title="Performance vs Satisfaction", opacity=0.8)
    fig1.update_layout(plot_bgcolor="white")

    # Productivity by Department
    fig2 = px.box(dff, x="Department", y="Productivity_Score", color="Department",
                  title="Productivity by Department")
    fig2.update_layout(plot_bgcolor="white")

    # Resignation by Department
    dept_resign = dff.groupby("Department")["Resigned"].mean().reset_index()
    fig3 = px.bar(dept_resign, x="Department", y="Resigned",
                  title="Resignation Rate by Department", color="Resigned")
    fig3.update_layout(plot_bgcolor="white")

    # Tenure vs Resignation
    fig4 = px.histogram(dff, x="Tenure_Years", color="Resigned",
                        nbins=30, title="Tenure Distribution by Resignation")
    fig4.update_layout(plot_bgcolor="white")

    # Correlation Matrix
    corr = dff[["Performance_Score", "Employee_Satisfaction_Score", "Productivity_Score",
                "Monthly_Salary", "Overtime_Hours", "Training_Hours", "Tenure_Years"]].corr()
    fig5 = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu")

    # Confusion Matrix
    fig6 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     title="Model Confusion Matrix (Logistic Regression)")
    fig6.update_xaxes(title="Predicted")
    fig6.update_yaxes(title="Actual")

    # Feature Importance
    fig7 = px.bar(feature_importance, x="Importance", y="Feature", orientation="h",
                  title="Top Predictive Features for Resignation", color="Importance",
                  color_continuous_scale="Viridis")
    fig7.update_layout(plot_bgcolor="white")

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7


# RUN SERVER
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)