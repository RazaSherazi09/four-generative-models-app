# Section 2.6: Experimental Analysis (E.1 – E.7)

## E.1 Masked Autoencoder: 75% Image Masking vs. 15% Text Masking
Natural language possesses high semantic density and low local redundancy: masking 15% of tokens in BERT removes discrete words whose identities cannot be recovered by simple local interpolation, forcing the Transformer to model syntactic and semantic dependencies. By contrast, natural and satellite images exhibit strong spatial redundancy—neighboring $8 \times 8$ patches in a $64 \times 64$ Sentinel-2 tile share smooth color gradients, textures, and structural continuity. Masking only 15% of image patches would allow a decoder to solve pixel reconstruction via trivial local boundary interpolation without learning high-level land-cover semantics.

By masking **75% of the image (48 of 64 patches)** and feeding only **16 scattered visible patches** to the encoder, the MAE eliminates short-range extrapolation shortcuts. In `results/figures/mae_reconstructions.png`, the model had to infer global spatial layouts from sparse visual cues: extending linear transportation corridors across masked gaps (`Highway`), continuing river bends and coastlines (`River`, `SeaLake`), preserving dense urban grid textures (`Residential`, `Industrial`), and distinguishing homogeneous canopy coverage (`Forest`) from parcel boundaries (`AnnualCrop`, `PermanentCrop`). While reconstructed patches appear smoothed due to pixel-wise MSE averaging over high-frequency uncertainty, the structural layout proves the encoder learned holistic scene-level representations.

---

## E.2 Few-Label EuroSAT Classification & Class Confusion Analysis
With only **16 labeled examples per class (160 total out of 16,200)** and a frozen linear probe (`nn.Linear(192, 10)`), the MAE representation achieved **49.52% test accuracy** across 5,400 held-out tiles (nearly $5\times$ the 10.00% random baseline). Inspecting `results/figures/mae_confusion_matrix.png` and `results/tables/mae_confusion_matrix.csv` reveals three primary confusion clusters driven by shared spectral and textural signatures at $64 \times 64$ resolution:

1. **Agricultural & Vegetation Classes (`AnnualCrop`, `PermanentCrop`, `HerbaceousVegetation`, `Pasture`):** These four classes share green-to-brown vegetated color palettes and lack sharp man-made edges. With 75% masking during pretraining, fine sub-patch crop-row periodicity is partially smoothed out, causing `PermanentCrop` and `HerbaceousVegetation` to overlap heavily.
2. **Linear Infrastructure vs. Waterways (`Highway` vs. `River`):** Both classes appear as narrow, elongated, continuous ribbons bisecting surrounding terrain. Without fine spectral band ratios (such as near-infrared water absorption), dark asphalt roads and narrow rivers share nearly identical geometric features.
3. **Built-Up Structures (`Industrial` vs. `Residential`):** Both contain high-frequency rectilinear building footprints and road networks, differing primarily in roof scale and spacing—distinctions that are subtle in low-resolution $8 \times 8$ patch embeddings trained with only 16 shots per class.

---

## E.3 VAE Training Dynamics & Effect of Removing the KL Term
Across 30 epochs on UHaT (`results/figures/vae_loss_terms.png` and `results/tables/vae_train_history.csv`), the two ELBO components evolved in opposite directions while their sum decreased monotonically:
* **Reconstruction Term ($-\log p_\theta(x|z)$):** Dropped rapidly from `202.13` nats (Epoch 1) to `105.87` nats (Epoch 30) on train (`106.44` nats on test) as the convolutional decoder learned to reconstruct 28×28 Urdu stroke geometries.
* **KL Divergence Term ($D_{\text{KL}}(q_\phi(z|x) \parallel \mathcal{N}(0, I))$):** Increased from `14.83` nats (Epoch 1) to `25.02` nats (Epoch 30) on train (`24.66` nats on test, or `~1.54` nats per latent dimension). Early in training, the encoder stayed close to the unit Gaussian prior; as optimization progressed, it allocated posterior capacity across all 16 latent dimensions to encode distinct character shapes and stroke slants without collapsing to zero.

**Effect of Removing the KL Term:** If $D_{\text{KL}}(q_\phi(z|x) \parallel \mathcal{N}(0, I))$ were removed from the objective, the model would degenerate into an unregularized deterministic autoencoder. The encoder would drive $\sigma \to 0$ and scatter posterior means $\mu(x)$ arbitrarily far apart in $\mathbb{R}^{16}$ with huge variance and dead gaps between clusters to maximize reconstruction fidelity. Consequently, sampling $z \sim \mathcal{N}(0, I)$ at test time would almost always land in empty, unvisited regions of latent space, producing fragmented, unrecognizable ink blobs rather than coherent Urdu characters.

---

## E.4 VAE vs. Diffusion Sample Sharpness
Comparing `results/figures/vae_prior_samples.png` (Part B) against `results/figures/ddpm_40x8_samples.png` (Part C) shows that **DDPM samples are substantially sharper, cleaner, and higher-contrast** than VAE samples:
* **Why VAE Samples Are Blurry:** The VAE optimizes a single-step Gaussian latent bottleneck ($d=16$) paired with a pixel-wise Bernoulli cross-entropy likelihood. Because a 16-dimensional continuous bottleneck cannot encode every high-frequency stroke edge and writer tremor, the decoder averages over all plausible stroke thicknesses and positions consistent with $z$, yielding soft, gray, anti-aliased edges. Moreover, unconditional prior sampling ($z \sim \mathcal{N}(0, I)$) occasionally lands in transition zones between character manifolds, blending two characters together.
* **Why DDPM Samples Are Sharper:** Instead of compressing an image into a low-dimensional bottleneck in one shot, DDPM maintains full $28 \times 28$ spatial resolution throughout a 1,000-step Markov chain and trains a multi-scale U-Net with self-attention (`AttnDownBlock2D` / `AttnUpBlock2D`) on $L_{\text{simple}} = \|\epsilon - \epsilon_\theta(x_t, t, y)\|^2$. Each reverse step removes a small increment of Gaussian noise conditioned on the exact class label $y$ and timestep $t$, iteratively snapping coarse stroke trajectories onto crisp, high-contrast foreground/background boundaries.

---

## E.5 DDPM Failure Modes & Visually Confusable Urdu Characters
While the class-conditional DDPM achieved high overall legibility (`38/40` and `35/40` in the blind human recognizability test, `results/figures/ddpm_40x8_samples.png`), errors concentrate on Urdu character groups that share identical base rasm (skeletons) and differ only by small diacritical dots (*nuqtas*), miniature *toay* markers, or subtle loop topology:
1. **Heh Variants (`haa2` vs. `haa3`):** Both represent phonetic /h/ forms (`ہ` vs. `ھ`) with compact looped strokes; at $28 \times 28$ resolution, slight variations in loop closure make `haa2` (`#25`) and `haa3` (`#30`) visually interchangeable.
2. **Dotted Alveolar/Retroflex Consonants (`zhaa` vs. `zaaa` vs. `zaal`):** `zaal` (`ذ`), `zaaa` (`ز`), and `zhaa` (`ژ`) share the exact same curved base stroke and differ only in having one versus three tiny dots above the curve. In `ddpm_blind_test_40.png` (`#27` and `#33`), faint or merged dots caused `zhaa` and `zaaa` to resemble one another.
3. **Ring-Headed & Bowl Consonants (`qaaf` vs. `faa`, `laam` vs. `noonghunna`):** When the terminal tail of `faa` (`ف`) curves slightly deeper or `qaaf` (`ق`) has one faint dot (`#26`), the two look similar; likewise, a shorter vertical stem on `laam` (`ل`, `#38`) resembles the deep dotless bowl of `noonghunna` (`ں`).

---

## E.6 BioASQ RAG Retrieval Failures & Generator Faithfulness
Across the first 100 test questions, dense retrieval with `all-MiniLM-L6-v2` missed all ground-truth relevant abstracts in the top 10 on **14 questions** (`results/tables/part_d_summary.json`). Three representative failure cases illustrate why dense bi-encoders miss specialized biomedical queries and how `Qwen2.5-1.5B-Instruct` behaves under missing context:

1. **QID 4213 — *"Is capmatinib effective for glioblastoma?"* (Relevant ID: `31776899`)**
   * **Why Retrieval Failed:** Specific drug-disease trial pairings require exact entity co-occurrence. MiniLM retrieved general MET-inhibitor and oncology abstracts (`32557339`, `18366283`) that mentioned solid tumors or other kinase inhibitors without containing the glioblastoma-specific capmatinib study.
   * **Generator Behavior (Hallucination):** Instead of abstaining, the generator hallucinated an affirmative answer (*"Yes, capmatinib is effective for glioblastoma. According to Passage [3], Capmatinib has been approved by the FDA..."*), conflating general oncology trials in Passage [3] with glioblastoma. This highlights why small 1.5B models can still hallucinate when distractors share surface keywords.

2. **QID 4456 — *"What is F105-P?"* (Relevant ID: `15908939`)**
   * **Why Retrieval Failed:** `F105-P` is an opaque alphanumeric identifier (an anti-HIV-1 gp120 monoclonal antibody variant) with zero lexical or morphological clues in its surface string. Without sparse BM25 exact-term matching, a 384-d dense embedding cannot map `F105-P` to HIV envelope glycoprotein abstracts.
   * **Generator Behavior (Faithful Abstention):** None of the top-5 passages mentioned `F105-P`, and the generator faithfully followed instructions by outputting: *"I do not know based on the provided passages."*

3. **QID 3807 — *"Which main viral protein is targeted by the drug remdesivir?"* (Relevant IDs: `30275474`, `29511076`)**
   * **Why Retrieval Failed:** Top retrieved abstracts (`34497279`, `33555378`) discussed remdesivir pharmacokinetics and drug-transporter interactions rather than its primary viral target (RNA-dependent RNA polymerase, RdRp).
   * **Generator Behavior (Supported by Distractor, Incorrect Scientifically):** Grounded strictly in Passage [1]'s transporter discussion, the model answered *"Remdesivir targets MRP4, OATP1B1/1B3, MATE1, and OCT1. [1]"*—which was factually supported by the retrieved passage yet scientifically wrong for the user's question due to upstream retrieval error.

---

## E.7 Why Recall@k Is Lower Than Hit@k for the Same k
By definition (Section 3.4), for a query $q$ with relevant abstract set $R_q$ and top-$k$ retrieved abstracts $\text{top}_k(q)$:

$$\text{Hit@}k(q) = \mathbb{I}\!\left[|\text{top}_k(q) \cap R_q| \ge 1\right], \qquad \text{Recall@}k(q) = \frac{|\text{top}_k(q) \cap R_q|}{\min(k, |R_q|)}$$

* **Hit@k is a binary step function:** It awards full credit (`1.0`) as soon as **at least one** relevant abstract appears anywhere in the top $k$, regardless of whether $R_q$ contains 1 relevant abstract or 20.
* **Recall@k measures fractional coverage:** Whenever a question has multiple relevant abstracts ($|R_q| > 1$) and $k > 1$, the denominator $\min(k, |R_q|)$ is strictly greater than 1. Finding 1 relevant abstract out of 5 gives $\text{Hit@5} = 1.0$ but $\text{Recall@5} = 1/5 = 0.20$.

In BioASQ, test questions have a **mean of 8.66 relevant abstracts per question** (`results/tables/part_d_summary.json`). Because most queries have multiple ground-truth abstracts, retrieving one or two relevant abstracts satisfies $\text{Hit@}k = 1$ while yielding partial coverage under $\text{Recall@}k$. Consequently, $\text{Recall@}k \le \text{Hit@}k$ holds mathematically for every query and every $k \ge 1$ (with equality only at $k = 1$, where $\min(1, |R_q|) = 1$ makes both metrics identical at `0.6223`, whereas at $k = 10$, $\text{Hit@10} = 0.8614$ vs. $\text{Recall@10} = 0.6024$).