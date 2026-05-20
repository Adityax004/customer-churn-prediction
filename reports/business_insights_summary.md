# Business Insights Summary

## Executive Summary

The Telco Customer Churn project is designed to help a telecom provider
identify customers who are likely to discontinue service and prioritize
retention actions before cancellation occurs. The workflow combines data
quality checks, exploratory analytics, feature engineering, tuned machine
learning models, model explainability, and segment-specific retention
recommendations.

The analysis focuses on customer demographics, service subscriptions,
account tenure, contract terms, billing behavior, monthly charges, total
charges, and churn labels.

## Business Impact of Churn

Customer churn affects the business through:

- Revenue loss from discontinued monthly recurring charges.
- Higher acquisition cost to replace lost customers.
- Lower customer lifetime value when customers leave early.
- Increased pressure on marketing and sales teams.
- Retention costs from discounts, incentives, and service interventions.

The goal is not only to maximize model accuracy. The more important business
goal is to catch as many true churners as possible while keeping retention
campaigns targeted enough to control cost.

## Key Churn Drivers to Validate

Typical churn patterns in the Telco dataset include:

- Month-to-month contracts usually carry higher churn risk than one-year or
  two-year contracts.
- Short-tenure customers are often less attached and more likely to churn.
- Electronic check payment tends to appear in higher-risk groups.
- Fiber optic customers can show higher churn when monthly charges are high
  or support add-ons are missing.
- Customers without online security or technical support are often more
  exposed to churn.
- High monthly charges can increase price sensitivity, especially for
  customers without long-term contracts.

## Why Recall Matters

Recall measures how many actual churners the model successfully identifies.
In churn prediction, a false negative means the company failed to intervene
with a customer who may cancel. That can result in full revenue loss and a
future reacquisition cost. A false positive may lead to an unnecessary
retention touch, but this cost is often lower than losing a customer.

For that reason, this project selects the best model using recall first and
ROC-AUC second.

## Recommended Retention Strategies

### New Customers

Segment trigger:

- Tenure of 0 to 12 months.

Recommended actions:

- Improve onboarding with first-bill support.
- Send product education journeys for internet and security services.
- Offer limited-time support check-ins before the first renewal decision.

### High Monthly Charge Users

Segment trigger:

- High-value customer flag or top quartile monthly charges.

Recommended actions:

- Provide bill review and plan-fit recommendations.
- Offer value-preserving bundles instead of broad discounts.
- Prioritize premium support callbacks for high-risk, high-revenue customers.

### Month-to-Month Contract Customers

Segment trigger:

- Contract equals month-to-month.

Recommended actions:

- Offer price-lock incentives for annual contracts.
- Use loyalty points or streaming/security bundles to increase stickiness.
- Provide personalized offers based on service usage and tenure.

### Senior Citizens

Segment trigger:

- SeniorCitizen equals Yes.

Recommended actions:

- Provide simplified support and clearer billing explanations.
- Use proactive service outreach instead of purely digital campaigns.
- Bundle technical support where friction or complexity may drive churn.

### Customers Without Online Security or Tech Support

Segment trigger:

- OnlineSecurity is not Yes or TechSupport is not Yes.

Recommended actions:

- Offer a free trial of security/support add-ons.
- Educate customers on the value of protection and support services.
- Monitor support tickets and service complaints as churn early-warning
  signals.

## Expected Business Outcomes

- Earlier identification of likely churners.
- Better retention campaign prioritization.
- Reduced avoidable churn in high-risk customer groups.
- Clearer understanding of which product, billing, and service factors drive
  churn.
- A repeatable analytics workflow suitable for dashboarding and operational
  monitoring.
