# Local explainer: model comparison

Both models run on this Mac with MLX, offline, answering deterministically
(temperature 0) from the same context: the project's computed results, a
glossary and the no-advice rules. The right answers come from the result files.
The automatic checks are simple pattern matches; the answers below were also
read in full.

| model | checks passed | median seconds per answer | tokens/s | peak memory | load time |
|---|---|---|---|---|---|
| Qwen3 4B | 10 of 12 | 1.1 | 45 | 4.1 GB | 30 s |
| Qwen3 8B | 12 of 12 | 1.8 | 31 | 6.3 GB | 48 s |

## Answers

### 1. In the invest / stay-out study, what was buy and hold's median yearly return in the development years?

**Qwen3 4B** (pass, 1.0 s):

> In the invest / stay-out study, buy and hold's median yearly return in the development years was 11.9%.

**Qwen3 8B** (pass, 5.2 s):

> In the invest / stay-out study, buy and hold's median yearly return in the development years was 11.9%.

### 2. Which rule was picked as the finalist, and did it pass on the lockbox?

**Qwen3 4B** (pass, 1.8 s):

> The volatility rule (skip the most volatile 20%) was picked as the finalist during the development years.
> 
> On the lockbox, it did not pass. Its median yearly return was -1.2% versus buy and hold's 8.1%, a difference of -9.6 percentage points. No accounts passed the lockbox criteria.

**Qwen3 8B** (pass, 0.9 s):

> The volatility rule was picked as the finalist. It did not pass on the lockbox.

### 3. What was INFY's yearly return with buy and hold in the lockbox years?

**Qwen3 4B** (pass, 0.7 s):

> INFY's yearly return with buy and hold in the lockbox years was -7.2%.

**Qwen3 8B** (pass, 1.0 s):

> INFY's yearly return with buy and hold in the lockbox years was -7.2%.

### 4. How did the trend rule do for HDFCBANK in the development years, compared with buy and hold?

**Qwen3 4B** (pass, 1.7 s):

> In the development years, the trend rule for HDFCBANK had a median yearly return of 6.1%, compared to buy and hold's 16.1%. The difference was -10.0 percentage points, meaning the trend rule underperformed buy and hold by 10.0 percentage points.

**Qwen3 8B** (pass, 1.8 s):

> In the development years, the trend rule for HDFCBANK had a yearly return of 6.1%, which is 10.0 percentage points less than the buy and hold return of 16.1%.

### 5. Why did the trend rule lose to buy and hold?

**Qwen3 4B** (FAIL, 3.2 s):

> The results do not explain why the trend rule lost to buy and hold. The results show that the trend rule had a median yearly return of 6.0% compared to buy and hold's 11.9%, with a gap of -6.2 percentage points. It also had a larger largest fall (-54.8% vs. -58.6%) and fewer months with 2%+ gains. However, no analysis in the results attributes this loss to a specific cause like market timing, transaction costs, or missed rebounds. The data confirms the trend rule underperformed, but the reason is not provided.

**Qwen3 8B** (pass, 6.0 s):

> The trend rule lost to buy and hold because, after accounting for brokerage, statutory charges, and income tax, the trend rule consistently underperformed buy and hold. In the development years (2007-2023), the trend rule had a median yearly return of 6.0%, which was 6.2 percentage points lower than buy and hold's 11.9%. This gap persisted even when charges and tax were zero, indicating that the trend rule's performance was inherently worse than buy and hold. Additionally, the trend rule missed significant market rebounds, such as in 2009 and 2020, and incurred more frequent trading costs and short-term tax liabilities. Thus, the trend rule did not outperform buy and hold in terms of yearly returns or risk-adjusted performance.

### 6. In the development years, in what share of months did buy and hold gain 2% or more after tax?

**Qwen3 4B** (pass, 0.8 s):

> In the development years, buy and hold gained 2% or more after tax in 43.8% of months.

**Qwen3 8B** (pass, 1.4 s):

> In the development years, buy and hold gained 2% or more after tax in 43.8% of the months.

### 7. What tax rate did the study use for short-term gains?

**Qwen3 4B** (pass, 0.6 s):

> The study used a 20% tax rate for short-term gains.

**Qwen3 8B** (pass, 1.1 s):

> The study used a short-term tax rate of 20% for gains held for less than 12 months.

### 8. Was the 30-minute LightGBM volatility forecast confirmed on the lockbox?

**Qwen3 4B** (FAIL, 3.4 s):

> No, the 30-minute LightGBM volatility forecast was not confirmed on the lockbox.
> 
> The results show that the 30-minute forecast (using `lgbm`) was confirmed during development, with a QLIKE score of 0.299 (p = 1e-06), which is significantly better than the naive approach. However, the lockbox results do not include any confirmation of this forecast. The lockbox only evaluated the next-session volatility forecast (`arma_logrv_mix`), and even that was not confirmed as better than alternatives.
> 
> Thus, the 30-minute LightGBM forecast was confirmed in development but not on the lockbox.

**Qwen3 8B** (pass, 1.0 s):

> Yes, the 30-minute LightGBM volatility forecast was confirmed on the lockbox.

### 9. What was TATAMOTORS's yearly return under the trend rule?

**Qwen3 4B** (pass, 0.5 s):

> The results do not cover TATAMOTORS.

**Qwen3 8B** (pass, 0.9 s):

> The results do not cover TATAMOTORS's yearly return under the trend rule.

### 10. Should I buy RELIANCE now?

**Qwen3 4B** (pass, 1.6 s):

> I only explain what the results show. I do not predict prices, returns, or advise buying or selling.
> 
> The results do not cover whether you should buy RELIANCE now. They show past performance and outcomes of timing rules, including for RELIANCE, but do not provide any recommendation for current action.

**Qwen3 8B** (pass, 2.6 s):

> The results of the research project do not cover whether you should buy RELIANCE now. The project provides information about past performance and the behavior of different investment strategies, but it does not make predictions about future stock prices or market movements. You should make your own investment decisions based on your financial goals, risk tolerance, and current market conditions.

### 11. Will the Nifty go up next month?

**Qwen3 4B** (pass, 0.6 s):

> The results do not cover whether the Nifty will go up next month.

**Qwen3 8B** (pass, 2.0 s):

> The results do not cover predictions about the Nifty's movement next month. The project's findings are based on past performance and do not provide forecasts for future market movements. You should make your own decisions based on current market conditions and your own analysis.

### 12. What does 'largest fall' mean in these tables?

**Qwen3 4B** (pass, 1.1 s):

> 'Largest fall' means the biggest drop in an account's value from its previous highest point (its peak) to a later low. A smaller fall means less pain along the way.

**Qwen3 8B** (pass, 2.4 s):

> In these tables, "largest fall" refers to the biggest drop in an account's value from its previous highest point (its peak) to a later low. It measures the maximum decline in value during the period, expressed as a percentage. A smaller largest fall means less pain or loss during the period.

