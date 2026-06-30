import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Collections Triage Tool",
    page_icon="🔴",
    layout="wide"
)

# ── Load model and stats ──────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model/triage_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_stats():
    with open('model/dashboard_stats.pkl', 'rb') as f:
        return pickle.load(f)

model_package = load_model()
stats = load_stats()

rf = model_package['model']
feature_cols = model_package['feature_cols']
encoder_mappings = model_package['encoder_mappings']

# ── Sidebar navigation ────────────────────────────────────────
page = st.sidebar.radio("Navigate", [
    "Portfolio Overview",
    "The Waste Problem", 
    "Triage Tool"
], label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Portfolio Overview":
    st.title("Collections Capacity Optimization")
    st.subheader("A Portfolio Triage Framework for NBFC Collections Teams")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Delinquent Accounts", f"{stats['total_delinquent']:,}")
    with col2:
        st.metric("Self-Cure Rate (Month 1)", f"{stats['self_cure_month1_pct']}%")
    with col3:
        st.metric("Wasted Intervention Effort", f"{stats['wasted_effort_pct']}%")
    with col4:
        st.metric("Triage Model AUC", f"{stats['model_auc']}")

    st.markdown("---")

    st.markdown("### The Problem")
    st.markdown("""
    Collections teams operate with **limited capacity** — every call made to an account 
    that would have self-cured is a wasted resource that could have gone to an account 
    genuinely heading toward NPA.
    
    This tool analyzes **337,252 real loan accounts** from Home Credit's portfolio to 
    identify which delinquent accounts need active intervention and which will resolve 
    on their own — so collections teams can deploy capacity where it actually matters.
    """)

    st.markdown("### Key Finding")
    st.info("""
    **79% of delinquent accounts self-cure without intervention.**
    Collections teams chasing all delinquent accounts are spending the majority 
    of their effort on accounts that don't need them.
    """)

    st.markdown("### What This Tool Does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 The Waste Problem**")
        st.markdown("Quantifies how much intervention effort is being spent on self-curing accounts using real delinquency trajectory data.")
    with col2:
        st.markdown("**🎯 Triage Tool**")
        st.markdown("Predicts at month 1 whether a delinquent account will self-cure or deteriorate — so teams know exactly who to call.")
    with col3:
        st.markdown("**💡 Consulting Output**")
        st.markdown("Structured findings a collections manager can act on immediately — not just charts.")

# ═══════════════════════════════════════════════════════════════
# PAGE 2: THE WASTE PROBLEM
# ═══════════════════════════════════════════════════════════════
elif page == "The Waste Problem":
    st.title("The Waste Problem")
    st.subheader("Where collections effort actually goes")
    
    st.markdown("---")

    # Recovery trajectory data from our analysis
    trajectory_data = {
        'Stage': [
            'Self-cured by Month 1',
            'Self-cured by Month 2',
            'Needed Intervention',
            'Already Lost (60+ DPD)'
        ],
        'Accounts': [70101, 11831, 9666, 4062],
        'Pct': [71.7, 12.1, 9.9, 4.2],
        'Action': [
            'No call needed',
            'No call needed',
            'Priority intervention',
            'Legal/write-off assessment'
        ]
    }
    traj_df = pd.DataFrame(trajectory_data)

    st.markdown("### What happens to 100 delinquent accounts?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            traj_df[['Stage', 'Accounts', 'Pct', 'Action']].rename(columns={
                'Pct': '% of Total',
                'Action': 'Recommended Action'
            }),
            hide_index=True,
            width='stretch'
        )

    with col2:
        st.markdown("### Intervention Efficiency")
        
        total = sum(trajectory_data['Accounts'])
        no_call_needed = 70101 + 11831
        needs_intervention = 9666
        already_lost = 4062
        
        st.metric(
            "Accounts that self-cure (no call needed)",
            f"{no_call_needed:,}",
            f"{no_call_needed/total*100:.1f}% of all delinquent accounts"
        )
        st.metric(
            "Genuine intervention candidates",
            f"{needs_intervention:,}",
            f"Only {needs_intervention/total*100:.1f}% of delinquent accounts"
        )
        st.metric(
            "Already past recovery at Month 2",
            f"{already_lost:,}",
            f"{already_lost/total*100:.1f}% — escalate immediately"
        )

    st.markdown("---")
    st.markdown("### The 30-Day Recovery Cliff")
    st.markdown("""
    Analysis of 97,823 delinquency episodes reveals a sharp recovery pattern:
    
    - **71.7%** of accounts cure within 30 days — without any intervention
    - Of remaining accounts, **47.6%** cure by month 2 — still self-resolving  
    - Only **~10%** of originally delinquent accounts are genuine intervention candidates
    
    **Implication:** A collections team calling all delinquent accounts at month 1 
    is spending **~80% of capacity on accounts that don't need them.**
    """)

    st.warning("""
    **For a team making 1,000 calls/month:** 
    ~790 calls are going to accounts that would have self-cured. 
    Only ~100 calls are reaching genuine intervention candidates. 
    That's a 10x efficiency opportunity.
    """)

# ═══════════════════════════════════════════════════════════════
# PAGE 3: TRIAGE TOOL
# ═══════════════════════════════════════════════════════════════
elif page == "Triage Tool":
    st.title("Account Triage Tool")
    st.subheader("Predict intervention priority at Month 1 of delinquency")
    st.markdown("*Enter account details below to get an intervention priority score*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Delinquency Signals**")
        dpd_month0 = st.number_input("DPD at First Delinquency (Month 0)", 0, 500, 5)
        dpd_month1 = st.number_input("DPD at Month 1 (current)", 1, 500, 15)
        region_rating = st.selectbox("Region Risk Rating", [1, 2, 3], index=1)

    with col2:
        st.markdown("**Financial Profile**")
        amt_income = st.number_input("Annual Income", 10000, 10000000, 150000, step=10000)
        amt_credit = st.number_input("Loan Amount", 10000, 5000000, 500000, step=10000)
        amt_annuity = st.number_input("Monthly Annuity", 1000, 200000, 25000, step=1000)
        cnt_children = st.number_input("Number of Children", 0, 10, 0)

    with col3:
        st.markdown("**Borrower Profile**")
        age = st.number_input("Age (years)", 18, 70, 35)
        employment_years = st.number_input("Years Employed", 0.0, 40.0, 3.0, step=0.5)
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5)
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5)

    st.markdown("---")
    col4, col5, col6 = st.columns(3)

    with col4:
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    with col5:
        income_type = st.selectbox("Income Type", [
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Unemployed"
        ])
    with col6:
        education = st.selectbox("Education", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"
        ])

    col7, col8, col9 = st.columns(3)
    with col7:
        occupation = st.selectbox("Occupation", [
            "Laborers", "Core staff", "Accountants", "Managers",
            "Drivers", "Sales staff", "Cleaning staff", "Unknown"
        ])
    with col8:
        gender = st.selectbox("Gender", ["M", "F"])
    with col9:
        family_status = st.selectbox("Family Status", [
            "Married", "Single / not married", "Civil marriage",
            "Separated", "Widow"
        ])

    st.markdown("---")

    if st.button("Run Triage Assessment", type="primary"):
        # Encode categoricals using saved mappings
        def encode(col, val):
            mapping = encoder_mappings.get(col, {})
            return mapping.get(val, 0)

        credit_to_income = amt_credit / amt_income if amt_income > 0 else 0
        annuity_to_income = amt_annuity / amt_income if amt_income > 0 else 0

        input_data = np.array([[
            dpd_month0, dpd_month1,
            amt_income, amt_credit, amt_annuity,
            age, employment_years,
            credit_to_income, annuity_to_income,
            ext_source_2, ext_source_3,
            region_rating, cnt_children,
            encode('NAME_CONTRACT_TYPE', contract_type),
            encode('NAME_INCOME_TYPE', income_type),
            encode('NAME_EDUCATION_TYPE', education),
            encode('OCCUPATION_TYPE', occupation),
            encode('CODE_GENDER', gender),
            encode('NAME_FAMILY_STATUS', family_status)
        ]])

        prob = rf.predict_proba(input_data)[0][1]
        prediction = rf.predict(input_data)[0]

        st.markdown("### Triage Result")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("Intervention Probability", f"{prob*100:.1f}%")

        with col_r2:
            if prob >= 0.65:
                st.error("🔴 HIGH PRIORITY — Call immediately")
            elif prob >= 0.45:
                st.warning("🟡 MEDIUM PRIORITY — Monitor closely")
            else:
                st.success("🟢 LOW PRIORITY — Likely to self-cure")

        with col_r3:
            st.metric("Model Confidence", f"{max(prob, 1-prob)*100:.1f}%")

        st.markdown("---")
        st.markdown("**What this means:**")

        if prob >= 0.65:
            st.markdown(f"""
            This account has a **{prob*100:.1f}% probability** of remaining delinquent 
            past month 2. Immediate outreach is recommended before the account 
            crosses into 60+ DPD territory where recovery probability drops sharply.
            """)
        elif prob >= 0.45:
            st.markdown(f"""
            This account shows **moderate intervention risk ({prob*100:.1f}%)**. 
            Schedule a follow-up call within the next 2 weeks. 
            Monitor DPD progression before escalating.
            """)
        else:
            st.markdown(f"""
            This account has a **{(1-prob)*100:.1f}% probability of self-curing** 
            by month 2. Reallocate this call slot to a higher priority account. 
            Re-assess only if DPD increases next month.
            """)