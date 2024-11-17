# Customer Segmentation

### **Cluster 2: Phone Service Only Customers**
**Churn Rate: 7.43%** (Lowest among all clusters)

Customers in Cluster 0 are moderate-tenure users who only subscribe to phone services. They tend to prefer traditional payment methods and longer-term contracts, indicating a possible preference for stability and familiarity.

**Characteristics:**

- **Tenure:** Medium (Mean: ~30.7 months, Median: 25 months)
- **Monthly Charges:** Low (Mean: ~$21.08, Median: ~$20.15)
- **Total Charges:** Low (Mean: ~$665.22, Median: ~$523.68)
- **Services Used:**
  - **Phone Subscription:** 100% have a phone subscription
  - **Internet Service:** 100% do not have internet service
  - **Multiple Lines:** 77.6% have a single line; 22.4% have multiple lines
- **Demographics:**
  - **Gender:** Evenly split between male and female
  - **Senior Citizens:** Predominantly not senior citizens (96.6% No)
  - **Partner Status:** Slightly more without partners (51.8% No)
  - **Dependents:** Majority without dependents (58.1% No)
- **Contract Type:**
  - **Two-Year Contract:** 41.6%
  - **Month-to-Month:** 34.5%
  - **One-Year Contract:** 23.9%
- **Billing and Payment:**
  - **Paperless Billing:** Majority do not use paperless billing (70.7% No)
  - **Payment Method:** Preference for mailed checks (48.4%), followed by bank transfers and credit card payments (both ~21.8%)

**Marketing Strategies:**

- **Upselling Internet Services:** Introduce attractive bundled packages that combine phone and internet services at a discounted rate to increase their monthly spending and enhance their engagement with more company offerings.
- **Promote Convenience Features:** Highlight the benefits of paperless billing and automatic payment methods to enhance their billing experience.
- **Loyalty Programs:** Implement loyalty rewards for their continued patronage, especially since a significant portion are on long-term contracts.

---

### **Cluster 1: Newer, Price-Sensitive Customers**
**Churn Rate: 44.25%** (Highest among all clusters)

Cluster 1 consists of newer customers who are possibly more price-sensitive and prefer flexibility, as indicated by their month-to-month contracts. They are less engaged with additional services and favor digital interactions.

**Characteristics:**

- **Tenure:** Short (Mean: ~15.3 months, Median: 12 months)
- **Monthly Charges:** Moderate (Mean: ~$68.60, Median: ~$70.85)
- **Total Charges:** Moderate (Mean: ~$1,043.98, Median: ~$795.65)
- **Services Used:**
  - **Phone Subscription:** 85% have a phone subscription
  - **Internet Service:** Mix of Fiber Optic (53.4%) and DSL (46.6%)
  - **Add-on Services:** Low adoption rates for online security, backup services, and device protection (over 70% have not subscribed)
- **Demographics:**
  - **Gender:** Evenly split between male and female
  - **Senior Citizens:** Higher proportion of senior citizens compared to Cluster 0 (19.9% Yes)
  - **Partner Status:** Majority without partners (67.3% No)
  - **Dependents:** Predominantly without dependents (80.1% No)
- **Contract Type:**
  - **Month-to-Month:** 87.4%
- **Billing and Payment:**
  - **Paperless Billing:** Majority use paperless billing (67.3% Yes)
  - **Payment Method:** Preference for electronic checks (49.7%)

**Marketing Strategies:**

- **Retention Efforts:** Given their short tenure and flexible contracts, they may have a higher risk of churn. Implement retention campaigns focusing on customer satisfaction and addressing pain points.
- **Promote Long-Term Contracts:** Offer incentives such as discounted rates or added benefits to encourage switching to longer-term contracts.
- **Upsell Add-on Services:** Educate them on the value of additional services like online security and device protection to enhance their experience and increase loyalty.
- **Personalized Communication:** Use targeted messaging that resonates with their digital preferences, highlighting convenience and value.

---

### **Cluster 0: Loyal, High-Value Customers**
**Churn Rate: 14.35%** (Moderate churn rate)

Cluster 2 represents the company's most loyal and valuable customers. They have been with the company for a long time, subscribe to multiple services, and are comfortable with digital and automated interactions.

**Characteristics:**

- **Tenure:** Long (Mean: ~57.8 months, Median: 60 months)
- **Monthly Charges:** High (Mean: ~$88.51, Median: ~$91.55)
- **Total Charges:** High (Mean: ~$5,109.89, Median: ~$4,983.05)
- **Services Used:**
  - **Phone Subscription:** 91.5% have a phone subscription
  - **Internet Service:** Mix of Fiber Optic (60.1%) and DSL (39.9%)
  - **Add-on Services:** High adoption rates for online security (55.7% Yes), backup services (67.7% Yes), device protection (70.4% Yes), and technical support (58.2% Yes)
- **Demographics:**
  - **Gender:** Evenly split between male and female
  - **Senior Citizens:** Approximately 19.6% are senior citizens
  - **Partner Status:** Majority have partners (70.2% Yes)
  - **Dependents:** 35.9% have dependents
- **Contract Type:**
  - **Two-Year Contract:** 42.9%
  - **One-Year Contract:** 33.9%
  - **Month-to-Month:** 23.2%
- **Billing and Payment:**
  - **Paperless Billing:** Majority use paperless billing (67.9% Yes)
  - **Payment Method:** Prefer automatic payments via bank transfer (32.6%) and credit card (31.2%)

**Marketing Strategies:**

- **Enhance Loyalty Programs:** Offer exclusive deals, early access to new services, or special recognition to reinforce their loyalty.
- **Cross-Selling Opportunities:** Introduce them to any new premium services or upgrades that could enhance their current packages.
- **Solicit Feedback:** Engage them in feedback programs to understand their needs better and make them feel valued.
- **Maintain Service Excellence:** Ensure high-quality customer service and support to keep satisfaction levels high.

---

### **Integrating Churn Predictions**

To effectively reduce churn, it's crucial to integrate churn probability predictions with the cluster profiles:

- **High Churn Risk (Cluster 1):**
  - Focus retention strategies here, as customers have low tenure and flexible contracts.
  - Monitor customer satisfaction closely and address issues promptly.
  - Use predictive analytics to identify early signs of churn (e.g., decreased usage patterns).

- **Moderate to Low Churn Risk (Clusters 0 and 2):**
  - **Cluster 0:** While they have medium tenure, the lack of multiple services presents an upselling opportunity to deepen their engagement.
  - **Cluster 2:** Despite low churn risk due to high tenure and satisfaction, continuous engagement is necessary to prevent competitors from enticing them away.
