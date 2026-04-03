import pandas as pd
import numpy as np

np.random.seed(42)
N = 200

first_names = [
    "Aarav","Priya","Rohan","Sneha","Arjun","Divya","Kiran","Meera",
    "Vikram","Ananya","Rahul","Pooja","Amit","Kavya","Suresh","Lakshmi",
    "Nikhil","Sanya","Ravi","Tanvi","Aditya","Ishaan","Nisha","Harsha",
    "Deepak","Swathi","Naveen","Revathi","Prasad","Sirisha"
]
last_names = [
    "Sharma","Reddy","Patel","Kumar","Singh","Nair","Iyer","Rao",
    "Verma","Gupta","Joshi","Pillai","Bose","Das","Chandra"
]

names = [
    f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
    for _ in range(N)
]

attendance     = np.clip(np.random.normal(75, 15, N), 10, 100)
study_hours    = np.clip(np.random.normal(4, 2, N), 0, 10)
prev_grade     = np.clip(np.random.normal(65, 15, N), 20, 100)
assignments    = np.clip(np.random.normal(7, 2, N), 0, 10).astype(int)
gender         = np.random.choice(["Male", "Female"], N)
parent_edu     = np.random.choice(
    ["None", "School", "Graduate", "Postgraduate"],
    N, p=[0.15, 0.30, 0.35, 0.20]
)
internet       = np.random.choice(["Yes", "No"], N, p=[0.72, 0.28])

final_grade = (
    0.30 * attendance +
    0.25 * (study_hours * 10) +
    0.25 * prev_grade +
    0.10 * (assignments * 10) +
    0.05 * (np.where(internet == "Yes", 5, 0)) +
    np.random.normal(0, 5, N)
)
final_grade = np.clip(final_grade, 0, 100).round(1)
performance_label = np.where(final_grade >= 50, "Pass", "Fail")

df = pd.DataFrame({
    "student_id"            : [f"STU{str(i+1).zfill(3)}" for i in range(N)],
    "name"                  : names,
    "age"                   : np.random.randint(15, 20, N),
    "gender"                : gender,
    "attendance_pct"        : attendance.round(1),
    "study_hours_per_day"   : study_hours.round(1),
    "prev_grade"            : prev_grade.round(1),
    "assignments_submitted" : assignments,
    "parent_education"      : parent_edu,
    "internet_access"       : internet,
    "final_grade"           : final_grade,
    "performance_label"     : performance_label
})

df.to_csv("data/students.csv", index=False)
print(f"✅ Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nClass distribution:")
print(df["performance_label"].value_counts())
print(f"\nSample data (first 5 rows):")
print(df.head())
