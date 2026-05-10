import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')

# ── Load Dataset ─────────────────────────────────────────────────
df = pd.read_csv('cleaned_loan_data.csv')

print("=" * 60)
print("         FINANCIAL LOAN DASHBOARD — EDA REPORT")
print("=" * 60)

# ── Create Missing Columns Safely ───────────────────────────────
if 'default_flag' not in df.columns:
    print("\n⚠ default_flag column not found.")
    print("Creating default_flag with default value 0")
    df['default_flag'] = 0

if 'risk_category' not in df.columns:
    df['risk_category'] = 'Unknown'

if 'income_band' not in df.columns:
    df['income_band'] = 'Unknown'

if 'loan_size_category' not in df.columns:
    df['loan_size_category'] = 'Unknown'

if 'purpose' not in df.columns:
    df['purpose'] = 'Unknown'

if 'home_ownership' not in df.columns:
    df['home_ownership'] = 'Unknown'

if 'years_in_current_job' not in df.columns:
    df['years_in_current_job'] = 'Unknown'

if 'dti_ratio' not in df.columns:
    df['dti_ratio'] = 0

# ── KPI Summary ─────────────────────────────────────────────────
total_loans = len(df)

total_amount = (
    df['current_loan_amount'].sum()
    if 'current_loan_amount' in df.columns else 0
)

default_rate = (
    df['default_flag'].mean() * 100
    if 'default_flag' in df.columns else 0
)

avg_credit_score = (
    df['credit_score'].mean()
    if 'credit_score' in df.columns else 0
)

avg_income = (
    df['annual_income'].mean()
    if 'annual_income' in df.columns else 0
)

avg_loan_amount = (
    df['current_loan_amount'].mean()
    if 'current_loan_amount' in df.columns else 0
)

avg_dti = (
    df['dti_ratio'].mean()
    if 'dti_ratio' in df.columns else 0
)

print(f"\n📌 KEY PERFORMANCE INDICATORS")
print(f"   Total Loans          : {total_loans:,}")
print(f"   Total Loan Amount    : ₹{total_amount:,.0f}")
print(f"   Default Rate         : {default_rate:.2f}%")
print(f"   Avg Credit Score     : {avg_credit_score:.0f}")
print(f"   Avg Annual Income    : ₹{avg_income:,.0f}")
print(f"   Avg Loan Amount      : ₹{avg_loan_amount:,.0f}")
print(f"   Avg Debt-to-Income   : {avg_dti:.2f}")

# ── Analysis 1: Risk Category ───────────────────────────────────
if 'risk_category' in df.columns:

    risk_analysis = df.groupby('risk_category').agg(
        total_loans=('default_flag', 'count'),
        defaults=('default_flag', 'sum'),
        avg_loan=('current_loan_amount', 'mean'),
        avg_income=('annual_income', 'mean')
    ).reset_index()

    risk_analysis['default_rate_pct'] = (
        risk_analysis['defaults'] /
        risk_analysis['total_loans'] * 100
    ).round(2)

else:
    risk_analysis = pd.DataFrame()

# ── Analysis 2: Loan Purpose ────────────────────────────────────
if 'purpose' in df.columns:

    purpose_analysis = df.groupby('purpose').agg(
        total_loans=('default_flag', 'count'),
        defaults=('default_flag', 'sum'),
        total_amount=('current_loan_amount', 'sum')
    ).reset_index()

    purpose_analysis['default_rate_pct'] = (
        purpose_analysis['defaults'] /
        purpose_analysis['total_loans'] * 100
    ).round(2)

    purpose_analysis = purpose_analysis.sort_values(
        'total_loans',
        ascending=False
    ).head(10)

else:
    purpose_analysis = pd.DataFrame()

# ── Analysis 3: Income Band ─────────────────────────────────────
if 'income_band' in df.columns:

    income_analysis = df.groupby('income_band').agg(
        total_loans=('default_flag', 'count'),
        defaults=('default_flag', 'sum'),
        avg_loan=('current_loan_amount', 'mean')
    ).reset_index()

    income_analysis['default_rate_pct'] = (
        income_analysis['defaults'] /
        income_analysis['total_loans'] * 100
    ).round(2)

else:
    income_analysis = pd.DataFrame()

# ── Analysis 4: Loan Size ───────────────────────────────────────
if 'loan_size_category' in df.columns:

    size_analysis = df.groupby('loan_size_category').agg(
        total_loans=('default_flag', 'count'),
        defaults=('default_flag', 'sum'),
        avg_amount=('current_loan_amount', 'mean')
    ).reset_index()

    size_analysis['default_rate_pct'] = (
        size_analysis['defaults'] /
        size_analysis['total_loans'] * 100
    ).round(2)

else:
    size_analysis = pd.DataFrame()

# ── Analysis 5: Home Ownership ──────────────────────────────────
if 'home_ownership' in df.columns:

    home_analysis = df.groupby('home_ownership').agg(
        total_loans=('default_flag', 'count'),
        defaults=('default_flag', 'sum')
    ).reset_index()

    home_analysis['default_rate_pct'] = (
        home_analysis['defaults'] /
        home_analysis['total_loans'] * 100
    ).round(2)

else:
    home_analysis = pd.DataFrame()

# ── Analysis 6: Employment Duration ─────────────────────────────
if 'years_in_current_job' in df.columns:

    job_analysis = df.groupby('years_in_current_job').agg(
        total_loans=('default_flag', 'count'),
        defaults=('default_flag', 'sum')
    ).reset_index()

    job_analysis['default_rate_pct'] = (
        job_analysis['defaults'] /
        job_analysis['total_loans'] * 100
    ).round(2)

else:
    job_analysis = pd.DataFrame()

# ── Save Summary CSVs ───────────────────────────────────────────
if not risk_analysis.empty:
    risk_analysis.to_csv('summary_risk.csv', index=False)

if not purpose_analysis.empty:
    purpose_analysis.to_csv('summary_purpose.csv', index=False)

if not income_analysis.empty:
    income_analysis.to_csv('summary_income.csv', index=False)

if not size_analysis.empty:
    size_analysis.to_csv('summary_size.csv', index=False)

print("\n✅ Summary CSVs saved for Power BI!")

# ═════════════════════════════════════════════════════════════════
# ENHANCED PROFESSIONAL DASHBOARD
# ═════════════════════════════════════════════════════════════════

plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 3, figsize=(22, 12))

fig.patch.set_facecolor('#f4f6f9')

fig.suptitle(
    'BANK LOAN ANALYTICS DASHBOARD',
    fontsize=24,
    fontweight='bold',
    color='#1d3557',
    y=0.98
)

# ────────────────────────────────────────────────────────────────
# Chart 1 — Risk Category
# ────────────────────────────────────────────────────────────────
ax1 = axes[0, 0]

if not risk_analysis.empty:

    bars1 = ax1.bar(
        risk_analysis['risk_category'],
        risk_analysis['total_loans'],
        color=['#2ecc71', '#f39c12', '#e74c3c'],
        edgecolor='black',
        linewidth=1
    )

    ax1.set_title(
        'Loans by Risk Category',
        fontsize=14,
        fontweight='bold'
    )

    ax1.set_ylabel('Loan Count')

    for bar in bars1:

        height = bar.get_height()

        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height + 30,
            f'{int(height):,}',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

# ────────────────────────────────────────────────────────────────
# Chart 2 — Loan Purpose
# ────────────────────────────────────────────────────────────────
ax2 = axes[0, 1]

if not purpose_analysis.empty:

    top8 = purpose_analysis.head(8)

    bars2 = ax2.barh(
        top8['purpose'],
        top8['total_loans'],
        color='#3498db',
        edgecolor='black'
    )

    ax2.set_title(
        'Top Loan Purposes',
        fontsize=14,
        fontweight='bold'
    )

    ax2.set_xlabel('Loan Count')

    for bar in bars2:

        width = bar.get_width()

        ax2.text(
            width + 20,
            bar.get_y() + bar.get_height()/2,
            f'{int(width):,}',
            va='center',
            fontsize=9
        )

# ────────────────────────────────────────────────────────────────
# Chart 3 — Income Band
# ────────────────────────────────────────────────────────────────
ax3 = axes[0, 2]

if not income_analysis.empty:

    bars3 = ax3.bar(
        income_analysis['income_band'],
        income_analysis['total_loans'],
        color=['#00b4d8', '#2ecc71', '#9b59b6', '#34495e'],
        edgecolor='black'
    )

    ax3.set_title(
        'Loans by Income Band',
        fontsize=14,
        fontweight='bold'
    )

    ax3.set_ylabel('Loan Count')

    ax3.tick_params(axis='x', rotation=10)

    for bar in bars3:

        height = bar.get_height()

        ax3.text(
            bar.get_x() + bar.get_width()/2,
            height + 30,
            f'{int(height):,}',
            ha='center',
            fontsize=9
        )

# ────────────────────────────────────────────────────────────────
# Chart 4 — Credit Score Distribution
# ────────────────────────────────────────────────────────────────
ax4 = axes[1, 0]

if 'credit_score' in df.columns:

    valid_scores = df[
        df['credit_score'].between(300, 850)
    ]['credit_score']

    if not valid_scores.empty:

        ax4.hist(
            valid_scores,
            bins=25,
            color='#1d4e89',
            edgecolor='white',
            alpha=0.9
        )

        mean_score = valid_scores.mean()

        ax4.axvline(
            mean_score,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Average: {mean_score:.0f}'
        )

        ax4.legend(fontsize=10)

ax4.set_title(
    'Credit Score Distribution',
    fontsize=14,
    fontweight='bold'
)

ax4.set_xlabel('Credit Score')
ax4.set_ylabel('Borrowers')

# ────────────────────────────────────────────────────────────────
# Chart 5 — Loan Size
# ────────────────────────────────────────────────────────────────
ax5 = axes[1, 1]

if not size_analysis.empty:

    bars5 = ax5.bar(
        size_analysis['loan_size_category'],
        size_analysis['total_loans'],
        color=['#90e0ef', '#48cae4', '#0077b6', '#023e8a'],
        edgecolor='black'
    )

    ax5.set_title(
        'Loan Size Distribution',
        fontsize=14,
        fontweight='bold'
    )

    ax5.set_ylabel('Loan Count')

    for bar in bars5:

        height = bar.get_height()

        ax5.text(
            bar.get_x() + bar.get_width()/2,
            height + 30,
            f'{int(height):,}',
            ha='center',
            fontsize=9
        )

# ────────────────────────────────────────────────────────────────
# Chart 6 — Home Ownership
# ────────────────────────────────────────────────────────────────
ax6 = axes[1, 2]

if 'home_ownership' in df.columns:

    home_grouped = df['home_ownership'].value_counts().head(5)

    colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd'
    ]

    explode = [0.03] * len(home_grouped)

    wedges, texts, autotexts = ax6.pie(
        home_grouped.values,
        labels=home_grouped.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 10}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

ax6.set_title(
    'Home Ownership Distribution',
    fontsize=14,
    fontweight='bold'
)

# ────────────────────────────────────────────────────────────────
# Styling
# ────────────────────────────────────────────────────────────────
for ax in axes.flat:

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ────────────────────────────────────────────────────────────────
# Layout & Save
# ────────────────────────────────────────────────────────────────
plt.subplots_adjust(
    top=0.90,
    hspace=0.35,
    wspace=0.25
)

plt.savefig(
    'enhanced_loan_dashboard.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n✅ Enhanced dashboard saved as enhanced_loan_dashboard.png")

# ── Key Insights ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("        💡 KEY INSIGHTS FOR RESUME / LINKEDIN")
print("=" * 60)

print(f"1. Overall default rate is {default_rate:.1f}%")

if not risk_analysis.empty:

    highest_risk = risk_analysis.sort_values(
        'total_loans',
        ascending=False
    ).iloc[0]

    print(
        f"2. '{highest_risk['risk_category']}' has "
        f"{highest_risk['total_loans']:,} loans"
    )

if not purpose_analysis.empty:

    top_purpose = purpose_analysis.iloc[0]

    print(
        f"3. '{top_purpose['purpose']}' is the most common loan purpose "
        f"with {top_purpose['total_loans']:,} loans"
    )

if not income_analysis.empty:

    top_income = income_analysis.sort_values(
        'total_loans',
        ascending=False
    ).iloc[0]

    print(
        f"4. '{top_income['income_band']}' borrowers represent the "
        f"largest customer segment"
    )