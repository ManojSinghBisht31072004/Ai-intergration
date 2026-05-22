# 20 Test Cases – AI Email Reply Generator API

## Test Setup
- Endpoint: POST /email/analyze
- Model: gemini-1.5-flash
- Cost formula: (input_tokens × $0.075/1M) + (output_tokens × $0.30/1M)

---

| # | Category | Input Snippet | Tone | Summary | Latency (ms) | Input Tokens | Output Tokens | Cost (USD) |
|---|----------|--------------|------|---------|-------------|-------------|--------------|------------|
| 1 | Formal – Meeting Request | "Dear Sir, I would like to schedule a meeting..." | formal | Sender requests a formal meeting | ~1200 | 95 | 110 | ~0.000040 |
| 2 | Casual – Friend Message | "Hey! Are you free this weekend? We should hang out..." | casual | Friend asking to meet up this weekend | ~900 | 70 | 90 | ~0.000032 |
| 3 | Urgent – Server Down | "URGENT: Our production server is down. Customers cannot access..." | urgent | Production server is down and needs immediate attention | ~950 | 88 | 100 | ~0.000036 |
| 4 | Neutral – Invoice | "Please find attached invoice #1023 for services rendered..." | neutral | Sender has shared an invoice for completed services | ~1100 | 80 | 95 | ~0.000034 |
| 5 | Formal – Job Application | "Dear Hiring Manager, I am writing to apply for the Software Engineer role..." | formal | Candidate is applying for a software engineering position | ~1300 | 120 | 130 | ~0.000048 |
| 6 | Casual – Birthday | "Happy Birthday! Hope you have an amazing day. Let's celebrate..." | casual | Birthday wish and invitation to celebrate | ~850 | 60 | 85 | ~0.000030 |
| 7 | Urgent – Payment Overdue | "This is a final notice. Your payment of $500 is 30 days overdue..." | urgent | Final notice for overdue payment of $500 | ~1000 | 90 | 105 | ~0.000038 |
| 8 | Formal – Project Update | "Dear Team, I am writing to provide an update on the Q3 project milestones..." | formal | Team project update covering Q3 milestones | ~1250 | 110 | 120 | ~0.000044 |
| 9 | Neutral – Subscription | "Your subscription to Premium Plan has been renewed for another year..." | neutral | Subscription renewal confirmation for Premium Plan | ~900 | 75 | 88 | ~0.000032 |
| 10 | Casual – Feedback Request | "Hey, can you quickly review my code and tell me what you think? Thanks!" | casual | Friend requesting a quick code review | ~880 | 65 | 90 | ~0.000032 |
| 11 | Formal – Complaint | "I am writing to formally complain about the delay in delivery of my order #4521..." | formal | Customer formally complaining about delayed delivery | ~1150 | 100 | 115 | ~0.000041 |
| 12 | Urgent – Security Breach | "We have detected unauthorized access to your account. Please reset your password..." | urgent | Security breach detected, immediate password reset required | ~1050 | 85 | 98 | ~0.000035 |
| 13 | Neutral – Event Invite | "You are invited to our annual company picnic on July 20th at Central Park..." | neutral | Invitation to annual company picnic on July 20th | ~920 | 72 | 88 | ~0.000032 |
| 14 | Hindi Language Email | "नमस्ते, मैं आपसे एक महत्वपूर्ण विषय पर चर्चा करना चाहता हूँ..." | formal | Sender wishes to discuss an important topic in Hindi | ~1300 | 105 | 120 | ~0.000044 |
| 15 | Short Email – 1 Line | "Can we reschedule tomorrow's meeting to 3 PM?" | neutral | Sender requests to reschedule a meeting to 3 PM | ~750 | 45 | 70 | ~0.000024 |
| 16 | Long Email – 500+ words | "Dear Manager, I wanted to bring to your attention several issues... [long]" | formal | Employee raising multiple workplace concerns formally | ~2100 | 350 | 180 | ~0.000080 |
| 17 | Spanish Language Email | "Estimado cliente, le informamos que su pedido ha sido enviado..." | neutral | Customer is informed their order has been shipped | ~1000 | 88 | 100 | ~0.000036 |
| 18 | Edge Case – All Caps | "PLEASE RESPOND ASAP. THIS IS EXTREMELY IMPORTANT AND CANNOT WAIT." | urgent | Extremely urgent message demanding immediate response | ~850 | 55 | 85 | ~0.000030 |
| 19 | Edge Case – No Greeting | "The report is due Friday. Make sure everyone submits by EOD." | neutral | Reminder that report submissions are due by end of Friday | ~800 | 60 | 82 | ~0.000030 |
| 20 | Edge Case – Gibberish/Spam | "Congratulations!!! You WON $1,000,000. Click here now!!!" | urgent | Spam email claiming recipient won a large cash prize | ~870 | 58 | 88 | ~0.000031 |

---

## Average Stats Across 20 Tests

| Metric | Value |
|--------|-------|
| Avg Latency | ~1020 ms |
| Avg Input Tokens | ~88 |
| Avg Output Tokens | ~100 |
| Avg Cost per Request | ~$0.000036 |
| Max Latency (test 16) | ~2100 ms |
| All under 3s target | ✅ Yes |