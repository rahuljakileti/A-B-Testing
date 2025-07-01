# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="A/B Testing Dashboard", layout="centered")

st.title("📊 A/B Testing Conversion Dashboard")

# --- File Upload or Manual Input ---
st.sidebar.header("📁 Step 1: Upload, Generate, or Enter Data")
input_method = st.sidebar.radio("Choose Data Source", ["Upload CSV File", "Generate Simulated CSV", "Manually Enter Data"])

if input_method == "Upload CSV File":
    file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.success("Data uploaded successfully!")
    else:
        st.warning("Please upload a CSV to proceed.")
        st.stop()
elif input_method == "Generate Simulated CSV":
    st.sidebar.markdown("Simulating data with these columns:")
    st.sidebar.code("['variant', 'converted', 'age', 'time_on_site', 'device']")
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'variant': np.random.choice(['A', 'B'], size=n),
        'age': np.random.randint(18, 60, size=n),
        'time_on_site': np.round(np.random.normal(loc=100, scale=20, size=n), 2),
        'device': np.random.choice(['Desktop', 'Mobile', 'Tablet', 'Other'], size=n)
    })
    df['converted'] = df.apply(lambda x: 1 if (x['variant'] == 'B' and np.random.rand() < 0.13)
                                or (x['variant'] == 'A' and np.random.rand() < 0.10) else 0, axis=1)
    st.success("Simulated data generated!")

    # Provide download link
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Simulated CSV",
        data=csv,
        file_name='simulated_ab_data.csv',
        mime='text/csv'
    )
else:
    st.sidebar.markdown("Enter a few rows of data below")
    example_csv = """variant,converted,age,time_on_site,device
A,0,25,95.4,Desktop
B,1,33,110.2,Mobile
A,1,40,120.6,Tablet
B,0,28,102.1,Other"""
    user_input = st.text_area("Paste your CSV data (first row must be headers):", example_csv, height=200)
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(user_input))
        st.success("Custom data parsed successfully!")
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        st.stop()

# --- Check Required Columns ---
required_cols = {'variant', 'converted', 'age', 'time_on_site', 'device'}
if not required_cols.issubset(df.columns):
    st.error("Dataset must contain the columns: variant, converted, age, time_on_site, device")
    st.stop()

# --- Basic Info ---
st.subheader("Data Preview")
st.dataframe(df.head())

# --- Conversion Rate by Variant ---
st.subheader("Conversion Rate by Variant")
conv_summary = df.groupby('variant')['converted'].mean().reset_index()
fig1, ax1 = plt.subplots()
sns.barplot(data=conv_summary, x='variant', y='converted', ax=ax1, hue='variant', dodge=False, palette='Set2', legend=False)
ax1.set_ylim(0, 0.15)
ax1.set_title("Conversion Rate: A vs B")
ax1.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig1)

# --- Conversion Rate by Device ---
st.subheader("Conversion Rate by Device Type")
device_summary = df.groupby('device')['converted'].mean().reset_index()
fig3, ax3 = plt.subplots()
sns.barplot(data=device_summary, x='device', y='converted', ax=ax3, palette='Set3')
ax3.set_title("Conversion Rate by Device")
ax3.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig3)

# --- Bootstrapping ---
st.subheader("Bootstrapped Confidence Interval (B - A)")
boot_diff = []
for _ in range(1000):
    sample = df.sample(frac=1, replace=True)
    conv_A = sample[sample['variant'] == 'A']['converted'].mean()
    conv_B = sample[sample['variant'] == 'B']['converted'].mean()
    boot_diff.append(conv_B - conv_A)

boot_diff = np.array(boot_diff)
lower, upper = np.percentile(boot_diff, [2.5, 97.5])

fig2, ax2 = plt.subplots()
sns.histplot(boot_diff, bins=50, kde=True, ax=ax2, color='skyblue')
ax2.axvline(lower, color='red', linestyle='--', label=f"Lower 95% CI: {lower:.4f}")
ax2.axvline(upper, color='green', linestyle='--', label=f"Upper 95% CI: {upper:.4f}")
ax2.axvline(0, color='black', linestyle='-', label="Zero Difference")
ax2.set_title("Bootstrapped CI for Conversion Rate Difference")
ax2.set_xlabel("Difference in Conversion (B - A)")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig2)

# --- Final Summary ---
st.subheader("📋 Final Summary")
conv_A = conv_summary.loc[conv_summary['variant']=='A', 'converted'].values[0]
conv_B = conv_summary.loc[conv_summary['variant']=='B', 'converted'].values[0]
st.write(f"- Variant A Conversion Rate: {conv_A:.4f}")
st.write(f"- Variant B Conversion Rate: {conv_B:.4f}")
st.write(f"- Bootstrapped 95% CI for (B - A): [{lower:.4f}, {upper:.4f}]")

# Explanation Section
st.markdown("### 🧠 Interpretation")
if lower > 0:
    st.success("✅ Variant B significantly outperforms Variant A.")
    st.markdown("- This likely means that the changes made in Variant B had a **positive impact** on conversion.")
    st.markdown("- Potential reasons: better layout, clearer call-to-action, faster load time, or better experience on certain devices.")
elif upper < 0:
    st.error("❌ Variant A outperforms Variant B.")
    st.markdown("- Variant B might have had a **negative impact**.")
    st.markdown("- Possible causes: confusing layout, slower load time, or users preferred the old design.")
else:
    st.warning("⚠️ No statistically significant difference found.")
    st.markdown("- The two variants are likely **performing similarly**, or the sample size isn't large enough.")
    st.markdown("- Try increasing the number of users or improving features to make effects clearer.")

st.markdown("---")
st.caption("Created using Streamlit ✨")
