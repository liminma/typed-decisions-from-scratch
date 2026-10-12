# 01 - Why typed decisions?

Code for Post 1: Why typed decisions?

Start the server first ([`00-setup/`](../00-setup/)). The server listens on `http://127.0.0.1:8080` by default. Set `LLAMA_URL` if it runs elsewhere. The scripts of Post 1 use Python's standard library only. Later posts will explain helper scripts `confidence.py`, `data.py`, `prompt.py` and `server.py`.

| Script | Shows |
|---|---|
| `free_text.py` | An open question: what does the customer want? |
| `label_list.py` | The list of 77 intent names is in the prompt. A request for one of them |
| `typed.py` | A typed decision: a label from the list with a confidence in one request |
| `throughput.py` | How many typed decisions per second with concurrency levels of 1, 2, 4 and 8 requests |

All output below is from a Dell Pro Max with GB10.

## 1. Prose

```bash
python3 01-why-typed-decisions/free_text.py
```

```
Customer: I am still waiting on my card?
Model:    It sounds like you are checking the status of your pending card delivery and inquiring about why it has not arrived yet.
          (25 tokens, 1045 ms on the server)

Customer: I need assistance with understanding which fiat currencies that you support.
Model:    As an AI assistant, I do not have access to real-time banking data or specific institutional policies. Therefore, I cannot provide the exact list of supported fiat currencies for your specific bank account.

To get the accurate and up-to-date information regarding which fiat currencies are supported (for deposits, withdrawals, or transfers), please:

1.  **Check your online banking portal:** Look under "Settings," "Account Details," or "Supported Currencies."
2.  **Review your account agreement:** This document usually lists the accepted currency codes (e.g., USD, EUR, GBP).

          (120 tokens, 4191 ms on the server)

Customer: How long is the delivery time to the US?
Model:    I'm sorry, but I don't have access to specific shipping details or current delivery times for your bank's services. For accurate information regarding delivery times to the US, please check the official website of the bank or contact their customer support directly. They can provide you with the most up-to-date and precise information based on your specific order or service type.
          (72 tokens, 2569 ms on the server)
```

## 2. A list of labels

Put BANKING77's 77 intent names in the prompt and ask for exactly one:

```bash
python3 01-why-typed-decisions/label_list.py
```

```
200 BANKING77 messages, the 77 intent names in the prompt, no grammar:
   199 replies were an exact label
     0 replies were a label inside other text
     1 replies were no label
   153 were the right label (76%)
  median time per request: 283 ms

  no label: 'Help me use American Express to add money to the account.' -> 'top_up_by_card'

Even when every reply is a label, nothing says how sure the model was of it.
```

## 3. A typed decision

```bash
python3 01-why-typed-decisions/typed.py
```

```
I am still waiting on my card?
  -> card_arrival  (code A-A, confidence 0.65, 234 ms)

I need assistance with understanding which fiat currencies that you support.
  -> fiat_currency_support  (code G-D, confidence 0.97, 228 ms)

How long is the delivery time to the US?
  -> card_delivery_estimate  (code A-B, confidence 0.65, 228 ms)

Help me use American Express to add money to the account.
  -> topping_up_by_card  (code E-I, confidence 0.35, 228 ms)
```

## 4. Concurrent requests

`throughput.py` sends 200 typed requests at 4 concurrent levels: 1, 2, 4 and 8 (the server has 8 slots):

```bash
python3 01-why-typed-decisions/throughput.py
```

```
Filled 8 slots with the system prompt in 17 s.

at once   requests/s   median ms   95th pct ms
      1         4.29         233           243
      2         4.44         454           467
      4         4.69         854           926
      8         4.68        1718          1840
```
