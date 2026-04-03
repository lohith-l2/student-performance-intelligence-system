# app.py — Student Performance Intelligence System
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os

from src.preprocessing import load_data, clean_data, encode_features, get_features_and_target
from src.predict        import predict_student
from src.model          import load_model

# ── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title = "Student Performance Intelligence System",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #2c3e50; text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem; color: #7f8c8d;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #3498db;
    }
    .pass-box {
        background: #d5f5e3; border-radius: 10px;
        padding: 1.5rem; text-align: center;
        border: 2px solid #2ecc71;
    }
    .fail-box {
        background: #fadbd8; border-radius: 10px;
        padding: 1.5rem; text-align: center;
        border: 2px solid #e74c3c;
    }
    .risk-high   { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low    { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Data Loader (cached) ──────────────────────────────────
@st.cache_data
def get_data():
    df     = load_data("data/students.csv")
    df     = clean_data(df)
    return df


# ── Sidebar Navigation ────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/graduation-cap.png",
    width=80
)
st.sidebar.title("🎓 SPIS")
st.sidebar.caption("Student Performance Intelligence System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home",
     "📊 Data Explorer",
     "📈 EDA & Insights",
     "🤖 Model Performance",
     "🔮 Predict Student",
     "⚠️  At-Risk Students",
     "➕ Add New Student"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Python & Streamlit")


# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-header">🎓 Student Performance Intelligence System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-powered dashboard to predict, analyze, and identify at-risk students</p>',
                unsafe_allow_html=True)

    df = get_data()

    # KPI Metrics
    total     = len(df)
    passing   = (df["performance_label"] == "Pass").sum()
    failing   = (df["performance_label"] == "Fail").sum()
    pass_rate = round(passing / total * 100, 1)
    avg_grade = round(df["final_grade"].mean(), 1)
    avg_att   = round(df["attendance_pct"].mean(), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Students",  total)
    c2.metric("✅ Passing",          passing)
    c3.metric("❌ Failing",          failing)
    c4.metric("📊 Pass Rate",        f"{pass_rate}%")
    c5.metric("📝 Avg Final Grade",  avg_grade)

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 About This Project")
        st.markdown("""
        This system uses **Machine Learning** to:
        - 📂 Analyze student performance data
        - 📊 Visualize key trends and patterns
        - 🤖 Predict Pass/Fail outcomes
        - ⚠️ Identify at-risk students early
        - 🔮 Make predictions for new students

        **Model:** Random Forest Classifier
        **Accuracy:** ~90% | **ROC-AUC:** ~0.92
        """)

    with col2:
        st.subheader("📁 Project Structure")
        st.code("""
student_performance_project/
├── app.py               ← Streamlit dashboard
├── train.py             ← Model training script
├── data/
│   └── students.csv     ← Dataset (200 students)
├── src/
│   ├── preprocessing.py ← Data cleaning
│   ├── model.py         ← ML model
│   ├── predict.py       ← Prediction engine
│   └── utils.py         ← EDA charts
└── models/
    └── model.pkl        ← Trained model
        """, language="")

    st.markdown("---")
    st.subheader("🚀 Quick Navigation")
    q1, q2, q3 = st.columns(3)
    q1.info("📊 **Data Explorer**\nBrowse and filter the raw dataset")
    q2.info("🔮 **Predict Student**\nEnter details and get instant prediction")
    q3.info("⚠️ **At-Risk Students**\nSee students needing intervention")


# ════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    df = get_data()

    # Filters
    st.subheader("🔍 Filter Data")
    f1, f2, f3 = st.columns(3)

    with f1:
        gender_filter = st.multiselect(
            "Gender",
            options=df["gender"].unique(),
            default=df["gender"].unique()
        )
    with f2:
        perf_filter = st.multiselect(
            "Performance",
            options=df["performance_label"].unique(),
            default=df["performance_label"].unique()
        )
    with f3:
        att_range = st.slider(
            "Attendance Range (%)",
            min_value=0, max_value=100,
            value=(0, 100)
        )

    # Apply filters
    filtered = df[
        (df["gender"].isin(gender_filter)) &
        (df["performance_label"].isin(perf_filter)) &
        (df["attendance_pct"].between(att_range[0], att_range[1]))
    ]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} students**")
    st.dataframe(filtered, use_container_width=True, height=400)

    # Download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Download Filtered Data",
        data      = csv,
        file_name = "filtered_students.csv",
        mime      = "text/csv"
    )

    # Summary stats
    st.markdown("---")
    st.subheader("📋 Summary Statistics")
    st.dataframe(
        filtered.describe().round(2),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════
# PAGE 3 — EDA & INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📈 EDA & Insights":
    st.title("📈 EDA & Insights")
    df = get_data()

    chart_files = {
        "Performance Distribution"    : "data/chart1_performance_distribution.png",
        "Attendance vs Final Grade"   : "data/chart2_attendance_vs_grade.png",
        "Study Hours: Pass vs Fail"   : "data/chart3_study_hours_boxplot.png",
        "Feature Correlation Heatmap" : "data/chart4_correlation_heatmap.png",
        "Parent Education Impact"     : "data/chart5_parent_education_impact.png",
        "At-Risk Student Analysis"    : "data/chart6_at_risk_students.png",
    }

    # Display charts in pairs
    chart_items = list(chart_files.items())
    for i in range(0, len(chart_items), 2):
        col1, col2 = st.columns(2)
        with col1:
            name, path = chart_items[i]
            st.subheader(name)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
        if i + 1 < len(chart_items):
            with col2:
                name, path = chart_items[i+1]
                st.subheader(name)
                if os.path.exists(path):
                    st.image(path, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    i1, i2, i3 = st.columns(3)
    i1.success("📈 Higher attendance strongly correlates with better final grades")
    i2.warning("📚 Failing students study ~2 fewer hours per day on average")
    i3.info("🎓 Students with graduate/postgraduate parents have higher pass rates")


# ════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Accuracy",       "90.00%")
    m2.metric("📈 ROC-AUC",        "0.9176")
    m3.metric("🔁 CV Mean",        "97.50%")
    m4.metric("📉 CV Std Dev",     "±3.06%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Confusion Matrix & Feature Importance")
        if os.path.exists("data/chart7_model_evaluation.png"):
            st.image("data/chart7_model_evaluation.png",
                     use_container_width=True)

    with col2:
        st.subheader("📋 Classification Report")
        report_data = {
            "Class"     : ["Fail", "Pass", "", "Accuracy", "Macro Avg", "Weighted Avg"],
            "Precision" : [0.86, 0.91, "", "", 0.88, 0.90],
            "Recall"    : [0.67, 0.97, "", 0.90, 0.82, 0.90],
            "F1-Score"  : [0.75, 0.94, "", 0.90, 0.84, 0.90],
            "Support"   : [9, 31, "", 40, 40, 40]
        }
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 Why Random Forest?")
        st.markdown("""
        - ✅ Handles non-linear relationships
        - ✅ Robust to outliers
        - ✅ Provides feature importances
        - ✅ Works well with small datasets
        - ✅ Built-in class balancing
        """)


# ════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT STUDENT
# ════════════════════════════════════════════════════════════
elif page == "🔮 Predict Student":
    st.title("🔮 Predict Student Performance")
    st.markdown("Fill in the student details below to get an instant prediction.")

    with st.form("prediction_form"):
        st.subheader("👤 Student Details")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age    = st.number_input("Age", min_value=14, max_value=25, value=17)
        with r1c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with r1c3:
            internet = st.selectbox("Internet Access", ["Yes", "No"])

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            attendance = st.slider("Attendance (%)",
                                   min_value=0, max_value=100, value=75)
        with r2c2:
            study_hours = st.slider("Study Hours / Day",
                                    min_value=0.0, max_value=10.0,
                                    value=4.0, step=0.5)

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            prev_grade = st.number_input("Previous Grade (0-100)",
                                          min_value=0.0, max_value=100.0, value=60.0)
        with r3c2:
            assignments = st.number_input("Assignments Submitted (0-10)",
                                           min_value=0, max_value=10, value=7)
        with r3c3:
            parent_edu = st.selectbox("Parent Education",
                                       ["None", "School", "Graduate", "Postgraduate"])

        submitted = st.form_submit_button("🔮 Predict Performance",
                                           use_container_width=True)

    if submitted:
        raw_input = {
            "age"                   : age,
            "gender"                : gender,
            "attendance_pct"        : attendance,
            "study_hours_per_day"   : study_hours,
            "prev_grade"            : prev_grade,
            "assignments_submitted" : assignments,
            "parent_education"      : parent_edu,
            "internet_access"       : internet
        }

        with st.spinner("Analyzing student profile..."):
            result = predict_student(raw_input)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        res1, res2, res3 = st.columns(3)

        if result["prediction"] == "Pass":
            res1.success(f"✅ Prediction: **PASS**")
        else:
            res1.error(f"❌ Prediction: **FAIL**")

        res2.metric("Pass Probability", f"{result['pass_prob']}%")
        res3.metric("Risk Level", result["risk_level"])

        # Probability bar
        st.markdown("---")
        st.subheader("📊 Probability Breakdown")
        prob_df = pd.DataFrame({
            "Outcome"     : ["Pass", "Fail"],
            "Probability" : [result["pass_prob"], result["fail_prob"]]
        })

        fig, ax = plt.subplots(figsize=(6, 2.5))
        colors = ["#2ecc71", "#e74c3c"]
        bars   = ax.barh(prob_df["Outcome"], prob_df["Probability"],
                         color=colors, edgecolor="white", height=0.5)
        for bar, val in zip(bars, prob_df["Probability"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontweight="bold")
        ax.set_xlim(0, 115)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Pass vs Fail Probability", fontweight="bold")
        st.pyplot(fig)
        plt.close()

        # Recommendation
        st.markdown("---")
        st.subheader("💡 Recommendation")
        if result["risk_level"] == "High Risk":
            st.error("""
            ⚠️ **Immediate Intervention Required**
            - Schedule counseling session
            - Assign a peer mentor
            - Monitor attendance weekly
            - Provide additional study resources
            """)
        elif result["risk_level"] == "Medium Risk":
            st.warning("""
            📋 **Monitor Closely**
            - Encourage increased study hours
            - Check in bi-weekly
            - Suggest study groups
            """)
        else:
            st.success("""
            🌟 **Student is on track!**
            - Continue current study habits
            - Consider advanced coursework
            - Encourage peer mentoring role
            """)


# ════════════════════════════════════════════════════════════
# PAGE 6 — AT-RISK STUDENTS
# ════════════════════════════════════════════════════════════
elif page == "⚠️  At-Risk Students":
    st.title("⚠️ At-Risk Student Identification")
    st.markdown("Students flagged based on attendance, study hours, and previous grades.")

    df = get_data()

    # Configurable thresholds
    st.subheader("⚙️ Risk Thresholds")
    t1, t2, t3 = st.columns(3)
    with t1:
        att_thresh   = st.slider("Max Attendance (%)", 40, 75, 60)
    with t2:
        study_thresh = st.slider("Max Study Hours",    1.0, 5.0, 3.0, step=0.5)
    with t3:
        grade_thresh = st.slider("Max Previous Grade", 30, 65, 50)

    # Identify at-risk
    at_risk = df[
        (df["attendance_pct"]      < att_thresh)   |
        (df["study_hours_per_day"] < study_thresh) |
        (df["prev_grade"]          < grade_thresh)
    ].copy()

    # Run batch predictions on at-risk students
    from src.predict import predict_batch, GENDER_MAP, INTERNET_MAP, EDU_MAP

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("⚠️ At-Risk Students", len(at_risk))
    r2.metric("✅ Safe Students",     len(df) - len(at_risk))
    r3.metric("📊 At-Risk Rate",      f"{len(at_risk)/len(df)*100:.1f}%")

    st.markdown("---")
    st.subheader(f"📋 At-Risk Student List ({len(at_risk)} students)")

    display_cols = ["name", "attendance_pct", "study_hours_per_day",
                    "prev_grade", "assignments_submitted", "performance_label"]

    # Color rows by performance
    def highlight_fail(row):
        color = "#fadbd8" if row["performance_label"] == "Fail" else "#fef9e7"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        at_risk[display_cols].style.apply(highlight_fail, axis=1),
        use_container_width=True,
        height=400
    )

    # Download at-risk list
    csv = at_risk[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Download At-Risk List",
        data      = csv,
        file_name = "at_risk_students.csv",
        mime      = "text/csv"
    )

    st.markdown("---")
    st.subheader("📊 At-Risk Analysis Charts")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Attendance distribution
    axes[0].hist(at_risk["attendance_pct"], bins=15,
                 color="#e74c3c", edgecolor="white", alpha=0.8)
    axes[0].set_title("Attendance Distribution (At-Risk)", fontweight="bold")
    axes[0].set_xlabel("Attendance (%)")
    axes[0].set_ylabel("Number of Students")

    # Study hours distribution
    axes[1].hist(at_risk["study_hours_per_day"], bins=15,
                 color="#f39c12", edgecolor="white", alpha=0.8)
    axes[1].set_title("Study Hours Distribution (At-Risk)", fontweight="bold")
    axes[1].set_xlabel("Study Hours / Day")
    axes[1].set_ylabel("Number of Students")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
# ════════════════════════════════════════════════════════════
# PAGE 7 — ADD NEW STUDENT
# ════════════════════════════════════════════════════════════
elif page == "➕ Add New Student":
    st.title("➕ Add New Student Data")
    st.markdown("Add a single student manually or upload a CSV file with multiple students.")

    tab1, tab2, tab3 = st.tabs([
        "✏️ Add Single Student",
        "📁 Upload CSV",
        "📋 View Template"
    ])

    # ── TAB 1: Manual Entry ───────────────────────────────
    with tab1:
        st.subheader("✏️ Add a Single Student")
        st.info("Fill in all fields below. The record will be appended to students.csv")

        with st.form("add_student_form"):
            r0c1, r0c2 = st.columns(2)
            with r0c1:
                new_id = st.text_input(
                    "Student ID",
                    value=f"STU{len(pd.read_csv('data/students.csv'))+1:03d}",
                    help="Unique ID e.g. STU201"
                )
            with r0c2:
                new_name = st.text_input("Full Name", placeholder="e.g. Priya Sharma")

            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                new_age    = st.number_input("Age", min_value=14, max_value=25, value=17)
            with r1c2:
                new_gender = st.selectbox("Gender", ["Male", "Female"])
            with r1c3:
                new_internet = st.selectbox("Internet Access", ["Yes", "No"])

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                new_attendance = st.slider("Attendance (%)", 0, 100, 75)
            with r2c2:
                new_study = st.slider("Study Hours / Day", 0.0, 10.0, 4.0, step=0.5)

            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                new_prev = st.number_input("Previous Grade (0-100)",
                                            min_value=0.0, max_value=100.0, value=60.0)
            with r3c2:
                new_assignments = st.number_input("Assignments Submitted (0-10)",
                                                   min_value=0, max_value=10, value=7)
            with r3c3:
                new_edu = st.selectbox("Parent Education",
                                        ["None", "School", "Graduate", "Postgraduate"])

            r4c1, r4c2 = st.columns(2)
            with r4c1:
                new_final = st.number_input("Final Grade (0-100)",
                                             min_value=0.0, max_value=100.0, value=55.0,
                                             help="Leave as 0 if not yet known")
            with r4c2:
                new_label = st.selectbox("Performance Label",
                                          ["Pass", "Fail"],
                                          help="Pass if final grade ≥ 50")

            submitted = st.form_submit_button("💾 Save Student", use_container_width=True)

        if submitted:
            if not new_name.strip():
                st.error("❌ Please enter the student's name.")
            elif not new_id.strip():
                st.error("❌ Please enter a Student ID.")
            else:
                # Build new row
                new_row = pd.DataFrame([{
                    "student_id"            : new_id.strip(),
                    "name"                  : new_name.strip(),
                    "age"                   : new_age,
                    "gender"                : new_gender,
                    "attendance_pct"        : new_attendance,
                    "study_hours_per_day"   : new_study,
                    "prev_grade"            : new_prev,
                    "assignments_submitted" : new_assignments,
                    "parent_education"      : new_edu,
                    "internet_access"       : new_internet,
                    "final_grade"           : new_final,
                    "performance_label"     : new_label
                }])

                # Check for duplicate ID
                existing = pd.read_csv("data/students.csv")
                if new_id.strip() in existing["student_id"].values:
                    st.error(f"❌ Student ID '{new_id}' already exists. Use a unique ID.")
                else:
                    # Append to CSV
                    new_row.to_csv("data/students.csv", mode="a",
                                   header=False, index=False)

                    # Clear cache so dashboard refreshes
                    st.cache_data.clear()

                    st.success(f"✅ Student **{new_name}** added successfully!")
                    st.balloons()

                    # Show the prediction for new student
                    st.subheader("🔮 Auto-Prediction for New Student")
                    raw_input = {
                        "age"                   : new_age,
                        "gender"                : new_gender,
                        "attendance_pct"        : new_attendance,
                        "study_hours_per_day"   : new_study,
                        "prev_grade"            : new_prev,
                        "assignments_submitted" : new_assignments,
                        "parent_education"      : new_edu,
                        "internet_access"       : new_internet
                    }
                    result = predict_student(raw_input)

                    p1, p2, p3 = st.columns(3)
                    if result["prediction"] == "Pass":
                        p1.success(f"✅ Predicted: **PASS**")
                    else:
                        p1.error(f"❌ Predicted: **FAIL**")
                    p2.metric("Pass Probability", f"{result['pass_prob']}%")
                    p3.metric("Risk Level", result["risk_level"])


    # ── TAB 2: Upload CSV ─────────────────────────────────
    with tab2:
        st.subheader("📁 Upload Multiple Students via CSV")
        st.info("Upload a CSV file with student records. It must match the required format.")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="File must contain correct column names"
        )

        REQUIRED_COLS = [
            "student_id", "name", "age", "gender",
            "attendance_pct", "study_hours_per_day", "prev_grade",
            "assignments_submitted", "parent_education",
            "internet_access", "final_grade", "performance_label"
        ]

        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.subheader("👀 Preview of Uploaded Data")
                st.dataframe(new_df.head(10), use_container_width=True)

                # Validate columns
                missing_cols = [c for c in REQUIRED_COLS if c not in new_df.columns]

                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                    st.info("Download the template from the 📋 View Template tab.")
                else:
                    # Check for duplicates
                    existing    = pd.read_csv("data/students.csv")
                    duplicate_ids = new_df[
                        new_df["student_id"].isin(existing["student_id"])
                    ]["student_id"].tolist()

                    if duplicate_ids:
                        st.warning(f"⚠️ Duplicate IDs found (will be skipped): {duplicate_ids}")
                        new_df = new_df[~new_df["student_id"].isin(duplicate_ids)]

                    st.success(f"✅ {len(new_df)} valid new records ready to import")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("New Records",   len(new_df))
                    with col2:
                        st.metric("Total After Import",
                                  len(existing) + len(new_df))

                    if st.button("💾 Import All Records", use_container_width=True):
                        new_df[REQUIRED_COLS].to_csv(
                            "data/students.csv", mode="a",
                            header=False, index=False
                        )
                        st.cache_data.clear()
                        st.success(f"✅ {len(new_df)} students imported successfully!")
                        st.balloons()

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")


    # ── TAB 3: Template ───────────────────────────────────
    with tab3:
        st.subheader("📋 CSV Template")
        st.markdown("""
        Your CSV must have exactly these columns in any order:

        | Column | Type | Valid Values |
        |--------|------|-------------|
        | student_id | Text | STU201, STU202... |
        | name | Text | Any name |
        | age | Integer | 14–25 |
        | gender | Text | Male / Female |
        | attendance_pct | Float | 0–100 |
        | study_hours_per_day | Float | 0–10 |
        | prev_grade | Float | 0–100 |
        | assignments_submitted | Integer | 0–10 |
        | parent_education | Text | None / School / Graduate / Postgraduate |
        | internet_access | Text | Yes / No |
        | final_grade | Float | 0–100 |
        | performance_label | Text | Pass / Fail |
        """)

        # Generate downloadable template
        template_df = pd.DataFrame([
            {
                "student_id"            : "STU201",
                "name"                  : "Example Student",
                "age"                   : 17,
                "gender"                : "Female",
                "attendance_pct"        : 80.0,
                "study_hours_per_day"   : 5.0,
                "prev_grade"            : 70.0,
                "assignments_submitted" : 8,
                "parent_education"      : "Graduate",
                "internet_access"       : "Yes",
                "final_grade"           : 72.0,
                "performance_label"     : "Pass"
            },
            {
                "student_id"            : "STU202",
                "name"                  : "Another Student",
                "age"                   : 16,
                "gender"                : "Male",
                "attendance_pct"        : 55.0,
                "study_hours_per_day"   : 2.0,
                "prev_grade"            : 42.0,
                "assignments_submitted" : 4,
                "parent_education"      : "School",
                "internet_access"       : "No",
                "final_grade"           : 44.0,
                "performance_label"     : "Fail"
            }
        ])

        st.dataframe(template_df, use_container_width=True)

        # Download template button
        csv_template = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download CSV Template",
            data      = csv_template,
            file_name = "student_template.csv",
            mime      = "text/csv"
        )