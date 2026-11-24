# Section 4: Experimental Results Tables

## 4.1 Table 1 – Data

| Dataset / Metric | Value |
| :--- | :--- |
| EuroSAT train / test images; labelled images used | 16,200 train / 5,400 test; 160 labelled (16 per class) |
| UHaT character classes; train / test images | 40 classes; 28,328 train / 4,747 test |
| BioASQ abstracts; chunks; test questions | 40,181 abstracts; 79,332 chunks; 707 test questions |
| Mean number of relevant abstracts per test question | 8.66 |

---

## 4.2 Table 2 – Training

| Model | Parameters | Final training loss | Time / GPU |
| :--- | :--- | :--- | :--- |
| MAE (encoder + decoder) | 3,173,248 | 0.0067 (Masked MSE) | 419.52s / Tesla T4 |
| VAE | 220,385 | 130.88 nats (-ELBO) | 19.31s / NVIDIA A10 |
| DDPM U-Net | 4,028,353 | 0.0225 ($L_{\text{simple}}$ MSE) | 864.47s / NVIDIA A10G |

---

## 4.3 Table 3 – Image results

| Metric | Result |
| :--- | :--- |
| MAE few-label classifier: test accuracy | 49.52% (2,674 / 5,400; chance = 10.00%) |
| VAE: test negative ELBO = reconstruction + KL (nats) | 131.1042 = 106.4449 + 24.6593 nats |
| Diffusion: blind test, member 1 / member 2 correct (of 40) | 38 / 35 (of 40) |
| Diffusion: images where both members agreed (of 40) | 37 (of 40) |

---

## 4.4 Table 4 – Retrieval

| Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.6223 | 0.8161 | 0.8614 | 0.6223 | 0.5863 | 0.6024 | 0.7061 |

---

## 4.5 Table 5 – Generation (100 questions)

| Metric | Result |
| :--- | :--- |
| ROUGE-L | 0.2275 |
| Citation precision | 0.6757 (75 / 111) |
| "I don't know" rate | 0.1600 (16 / 100) |
| Human: supported, member 1 / member 2 (of 20) | 16 / 15 (of 20) |
| Human: correct, member 1 / member 2 (of 20) | 16 / 13 (of 20) |