# Collections Capacity Optimization — A Triage Framework for Distressed Loan Portfolios

A data-driven triage system that identifies which delinquent loan accounts genuinely need collections intervention — and which will resolve on their own — built on real loan performance data from a 337,252-account NBFC-style portfolio.

## The Problem

Collections teams operate with limited capacity. Every call made to an account that would have self-cured is a wasted resource that could have gone to an account genuinely heading toward NPA. Most collections workflows treat every delinquent account the same way — call everyone, prioritize nothing.

This project asks a sharper question: **at the moment an account first goes delinquent, can you predict which way it will go — and how much capacity is currently being wasted by not knowing?**

## Key Finding

Analysis of 97,823 delinquency episodes across the portfolio reveals a sharp recovery pattern:

- **71.7%** of accounts cure within 30 days — with no intervention at all
- Of the remainder, another **47.6%** cure by month 2
- Only **~10%** of accounts that ever go delinquent are genuine intervention candidates
- That ~10% represents **₹84.4 Cr** in at-risk value, out of **₹176.7 Cr** in total portfolio exposure analyzed

**Implication:** a collections team calling every delinquent account at month 1 is spending roughly **79% of its effort on accounts that don't need it.**

## Approach

**1. Delinquency trajectory mapping**
Using monthly DPD (days-past-due) snapshots, every account's first delinquency event was traced forward to see whether it cured by month 1, cured by month 2, or deteriorated further. This produced the 30-day recovery cliff finding above.

**2. Intervention candidate definition**
Accounts still delinquent at month 1 (34,110 accounts) were treated as the real decision point — these are the accounts a collections team would actually be deciding whether to call. The target variable is whether the account cures by month 2 or remains/worsens.

**3. Triage classifier**
A Random Forest model was trained on delinquency severity (DPD at month 0 and month 1), borrower financial profile (income, credit-to-income ratio, annuity burden), and external credit signals (EXT_SOURCE scores) to predict intervention need.

- **ROC-AUC: 0.705**
- Balanced target distribution (47.5% / 52.5%), no resampling needed
- Top predictive features: DPD progression, age, external credit scores, debt-to-income ratios — behavioral and financial stress signals dominate over static demographic profile

**4. Interactive triage tool**
A three-page Streamlit application translates the analysis into a usable interface: portfolio-level findings, the waste-problem breakdown, and a live triage scorer where account details can be entered to get an intervention priority score (Low / Medium / High) with a recommended action.

## Data

Built on Home Credit's loan performance data (`POS_CASH_balance.csv`, `application_train.csv`) — 337,252 unique accounts, 10M+ monthly balance records. No synthetic data. The only constructed element is the intervention-priority threshold logic, which is a deliberate business design decision, not a data limitation: the HIGH priority threshold is set above 65% (not 50%) because in collections, missing a genuine NPA is more costly than an unnecessary call to a self-curing account.

## Tech Stack

- **Python** — pandas, numpy for data pipeline
- **scikit-learn** — Random Forest classifier
- **Streamlit** — interactive triage interface

## Project Structure
