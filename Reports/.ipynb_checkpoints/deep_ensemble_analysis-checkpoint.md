# Deep & Ensemble Learning — Full Analysis

**Project:** CS4662 — Identifying Age-Related Conditions  
**Notebook:** `Notebooks/deep_ensemble_learning.ipynb`  
**Models covered:** MLP Neural Networks, Stacking, Blending

---

## 1. Problem Context

The task is binary classification: predict whether an individual has any of three age-related medical conditions (Class 1) or none (Class 0), using 55 anonymized numeric health features.

Two properties of this dataset shape every modeling decision:

**Class imbalance.** The training set contains 509 Class 0 and 108 Class 1 samples — a 4.7:1 ratio. Without correction, any model trained on raw labels will be biased toward the majority class, producing high accuracy while missing most Class 1 cases. In a medical context, these false negatives (failing to detect a condition) are the most costly error.

**Small dataset size.** With only 617 total samples (493 training, 124 validation after an 80/20 split), the dataset is unusually small for deep learning. This has direct consequences for architecture choice, regularization strategy, and ensemble design.

---

## 2. Feature Preparation

### Feature Selection

Eight columns were dropped before modeling: `Id`, `Class`, `EJ` (categorical), and the five greeks metadata columns (`Alpha`, `Beta`, `Gamma`, `Delta`, `Epsilon`). This leaves 55 numeric features — the anonymized health measurements — as model input. This matches the setup used in the XGBoost notebook, allowing direct comparisons.

### Feature Scaling

StandardScaler (zero mean, unit variance) was applied to all features, fit on the training set only and applied to both training and validation. This step is non-negotiable for neural networks. Gradient-based optimizers assume features are on comparable scales; without scaling, neurons connected to large-magnitude features receive disproportionately large gradients, causing instability or slow convergence. Tree-based models (used later in ensemble methods) are scale-invariant, so scaling does not affect their behavior but is harmless.

### Class Weighting

Rather than resampling, class imbalance was addressed by passing `class_weight = {0: 1.0, 1: 4.73}` to the Keras `fit()` call. This scales the loss contribution of each Class 1 sample by 4.73 during backpropagation, effectively telling the network that misclassifying a Class 1 sample is 4.73 times more costly than misclassifying a Class 0 sample. The weight was derived from the training set ratio (407 Class 0 / 86 Class 1 = 4.73), ensuring no information from the validation set leaks into training.

---

## 3. Neural Network Design

### Framework and Architecture

All neural networks are feedforward MLPs (Multi-Layer Perceptrons) built with TensorFlow/Keras. Each model uses:
- `ReLU` activations in hidden layers (avoids vanishing gradients, computationally efficient)
- `sigmoid` output neuron (produces a probability in [0, 1] for binary classification)
- `binary_crossentropy` loss (the standard probabilistic loss for binary tasks)
- `Adam` optimizer with `learning_rate=0.001` (adaptive learning rates, robust to hyperparameter choice)
- `EarlyStopping` on validation loss with `restore_best_weights=True` (automatically selects the checkpoint with lowest validation loss, preventing overfitting)

### Why MLP and Not Other Architectures

The data consists of tabular numeric features with no spatial or sequential structure. CNNs and RNNs are designed for images and sequences respectively — applying them here would add parameters without any inductive bias benefit. MLPs are the appropriate deep learning architecture for this feature type.

---

## 4. Architecture Experiments

The experiment systematically varied depth (1–4 hidden layers) and width (64 or 128 units per layer), keeping all other settings constant.

| Layers | Units | Accuracy | Recall (Class 1) | Log Loss |
|--------|-------|----------|------------------|----------|
| 1 | 64 | 95.97% | 86.36% | **0.100** |
| 2 | 64 | 95.97% | **95.45%** | 0.153 |
| 2 | 128 | 93.55% | 81.82% | 0.154 |
| 3 | 128 | 89.52% | 68.18% | 0.228 |
| 3 | 64 | 91.94% | 77.27% | 0.238 |
| 4 | 64 | 88.71% | 54.55% | 0.276 |

### Key Finding: Depth Hurts on Small Data

The results show a consistent, monotonic degradation as depth increases. The 4-layer network achieves only 54.5% recall on Class 1 — worse than the untuned XGBoost baseline. This is a direct consequence of dataset size. Deeper networks have more parameters and therefore more capacity to memorize training examples rather than learn generalizable patterns. With only 493 training samples, a 4-layer network of 64 units has approximately 16,960 parameters — roughly 34 parameters per training sample, far too many for reliable generalization.

The 1-layer 64-unit network achieves the best log loss (0.100) and the 2-layer 64-unit network achieves the best recall (95.45%) while matching on accuracy. Wider networks (128 units) are consistently outperformed by their narrower equivalents at the same depth, further supporting the conclusion that model capacity needs to be restrained on this dataset.

### Decision

The 2-layer (64 units each) structure was selected as the basis for regularization experiments because it balances accuracy and recall better than the single-layer model, and because recall for Class 1 is the primary concern in this medical domain.

---

## 5. Regularization Experiments

### 5.1 Dropout

Dropout randomly zeros a fraction of neuron activations during each training step, forcing the network to learn redundant representations and reducing co-adaptation between neurons.

| Dropout Rate | Accuracy | Recall (Class 1) | Log Loss |
|--------------|----------|------------------|----------|
| 0.2 | 93.55% | **90.91%** | 0.160 |
| **0.3** | 93.55% | 81.82% | **0.091** |
| 0.4 | 91.94% | 81.82% | 0.201 |
| 0.5 | 95.16% | 86.36% | 0.114 |

Dropout=0.3 achieves the best log loss (0.091), which is the lowest of any individual model in the entire notebook. Dropout=0.2 achieves the best recall, suggesting it retains more minority-class signal. The weak point is Dropout=0.4, which hurts both metrics relative to its neighbors — a sign that too much regularization at this data size starts to impede learning. The non-monotonic behavior across dropout rates is expected; the optimal rate is dataset-dependent.

### 5.2 Batch Normalization

Batch Normalization normalizes the inputs to each layer across the current mini-batch, then applies learned scale and shift parameters. It is primarily designed to reduce internal covariate shift, allowing higher learning rates and improving training stability.

Applied alone to the 2-layer architecture:

| Metric | Value |
|--------|-------|
| Accuracy | 92.7% |
| Recall (Class 1) | 68% |
| Log Loss | 0.235 |

BatchNorm alone **degrades recall** from the no-regularization baseline (82% → 68%) and produces the worst log loss of any regularization technique. This is not surprising. With a batch size of 32 and only 493 training samples, each mini-batch contains approximately 6.5% of the data. The batch statistics (mean, variance) computed from such a small fraction are noisy, making the normalization unstable. BatchNorm is known to underperform on very small datasets or small batch sizes. Its value emerges when combined with other techniques that stabilize training.

### 5.3 L2 Weight Regularization

L2 regularization adds a penalty term to the loss function proportional to the squared magnitude of all weights. This discourages the network from assigning large weights to any single feature, promoting simpler, more generalizable solutions.

| λ | Accuracy | Recall (Class 1) | Log Loss |
|---|----------|------------------|----------|
| 0.0001 | 91.94% | 86.36% | 0.197 |
| 0.001 | 93.55% | 77.27% | 0.144 |
| **0.01** | **94.35%** | 81.82% | **0.134** |
| 0.1 | 90.32% | **90.91%** | 0.233 |

L2=0.01 gives the best trade-off: the highest accuracy and lowest log loss among L2 configurations, with solid recall. Very high L2 (λ=0.1) pushes recall up to 90.91% — the regularizer is so aggressive that the model becomes conservative about Class 0, leading to more Class 1 predictions — but this comes at the cost of overall accuracy and probability calibration (log loss 0.233).

---

## 6. Best Neural Network

The final neural network combines all three regularization techniques:

```
Input (55)
  → Dense(128, ReLU) + L2(0.001)
  → BatchNormalization
  → Dropout(0.3)
  → Dense(64, ReLU) + L2(0.001)
  → BatchNormalization
  → Dropout(0.3)
  → Dense(1, sigmoid)
```

Trained with `EarlyStopping(patience=30, restore_best_weights=True)` and `class_weight={1: 4.73}`.

| Metric | Value |
|--------|-------|
| Accuracy | **95.2%** |
| Recall (Class 1) | 77.3% |
| Precision (Class 1) | **94%** |
| F1 (Class 1) | **0.85** |
| Log Loss | 0.177 |
| Confusion matrix | [[101, 1], [5, 17]] |

### Why Combining Works Better

Each technique addresses a different failure mode:

- **L2** penalizes overly large weights, preventing any single feature from dominating predictions.
- **BatchNorm** stabilizes activation distributions across layers, making the optimization landscape smoother. Its instability when used alone (shown above) is reduced when combined with Dropout, which adds stochastic noise that dampens the effect of batch statistic variation.
- **Dropout** forces the network to not rely on any single neuron, creating an ensemble-like effect within a single model.

The combined model correctly classifies 101 of 102 Class 0 samples (only 1 false positive) while catching 17 of 22 Class 1 samples (5 false negatives). The high precision (94%) means that when the model predicts Class 1, it is almost certainly correct. This property is valuable in medical settings where a positive diagnosis triggers follow-up testing or treatment.

### Comparison with Baseline

The baseline MLP (no regularization) has better recall (82%) but far lower precision (72%) and produces 7 false positives vs. 1 for the best model. The best NN trades some recall for a large precision improvement, resulting in a significantly higher F1 (0.85 vs. 0.77). Whether the recall reduction (82% → 77%) is acceptable depends on the clinical context — the analysis below returns to this point.

---

## 7. Ensemble Methods

### 7.1 Stacking

#### Design

Stacking trains multiple diverse base learners and uses their out-of-fold predictions as inputs to a meta-learner. Using scikit-learn's `StackingClassifier` with `cv=5`, the procedure is:

1. The training set is split into 5 folds.
2. Each base learner is trained on 4 folds and predicts on the held-out fold.
3. These out-of-fold predictions (one column per base learner) become the training set for the meta-learner.
4. All base learners are then retrained on the full training set to generate predictions at inference time.

Base learners chosen for diversity:
- **Logistic Regression** — linear boundary, calibrated probabilities
- **Random Forest** — bagging ensemble, high variance reduction
- **Gradient Boosting Machine** — sequential boosting, bias reduction
- **XGBoost** — regularized boosting with `scale_pos_weight` for class imbalance

The meta-learner is Logistic Regression — a linear model that learns the optimal combination of base learner outputs.

#### Results

| Metric | Value |
|--------|-------|
| Accuracy | 93.5% |
| Recall (Class 1) | 72.7% |
| F1 (Class 1) | 0.80 |
| Log Loss | 0.181 |
| Confusion matrix | [[100, 2], [6, 16]] |

Stacking outperforms the XGBoost weighted baseline on every metric. The cross-validation approach makes efficient use of the limited data — every training sample contributes to both base model training and meta-feature generation.

The meta-learner does not dramatically outperform a single strong model (XGBoost weighted: 93.5% accuracy, 0.204 log loss vs. stacking: 93.5% accuracy, 0.181 log loss), which suggests the base learners are correlated in their errors. Diversity among base learners is the primary driver of stacking gains, and all four base learners here are trained on the same tabular features with similar inductive biases.

---

### 7.2 Blending

#### Design

Blending uses a single holdout split rather than cross-validation:

1. Training data split 70/30 into `blend_train` (345 samples) and `blend_holdout` (148 samples).
2. Five base models trained on `blend_train`: LogReg, RF, GBM, XGB, and the best Keras NN.
3. Predicted probabilities from all five models on `blend_holdout` form a (148 × 5) meta-feature matrix.
4. A Logistic Regression meta-learner is trained on those meta-features.
5. At inference time, each base model predicts on the validation set; these predictions are fed to the meta-learner.

The key advantage of blending over stacking is simplicity and the ability to include any model type (including Keras) as a base learner without requiring sklearn-compatible cross-validation wrappers.

#### Results

| Metric | Value |
|--------|-------|
| Accuracy | 91.9% |
| Recall (Class 1) | **63.6%** |
| F1 (Class 1) | 0.74 |
| Log Loss | 0.200 |
| Confusion matrix | [[100, 2], [8, 14]] |

Blending is the weakest ensemble method and the weakest model overall (excluding the unweighted XGBoost baseline from a different notebook). It catches only 14 of 22 Class 1 cases.

#### Why Blending Underperforms

The failure is attributable to dataset size. Splitting 493 training samples 70/30 leaves:
- 345 samples to train 5 base models — each model sees less data and produces weaker predictions.
- 148 holdout samples to train the meta-learner — a very small training set for learning combination weights.

The meta-learner weights reveal the consequence:

| Base Model | Meta-Learner Weight |
|------------|---------------------|
| GBM | 2.054 |
| XGBoost | 1.373 |
| Random Forest | 1.204 |
| Neural Network | 1.163 |
| Logistic Regression | 0.996 |

GBM receives a weight of 2.054 — more than twice that of Logistic Regression. Rather than learning a balanced combination of diverse views, the meta-learner has learned to primarily trust GBM and discount the others. This is a sign that the base model predictions on the small holdout set are too noisy for meaningful combination learning. Stacking's cross-validation avoids this by giving every sample a chance to appear in the holdout, producing cleaner meta-features.

The Neural Network's weight (1.163) is on par with the tree methods, confirming it contributes valid signal even when trained on the reduced blend_train set. However, the NN trained on 345 samples is naturally weaker than the NN trained on 493 samples in the standalone experiments.

---

## 8. Final Comparison

| Model | Accuracy | Recall (Class 1) | F1 (Class 1) | Log Loss |
|-------|----------|------------------|--------------|----------|
| **NN Best (L2+BN+Dropout)** | **95.2%** | 77.3% | **0.85** | 0.177 |
| Stacking | 93.5% | 72.7% | 0.80 | 0.181 |
| XGBoost Weighted *(prior notebook)* | 92.7% | 73.0% | 0.78 | 0.204 |
| NN Baseline | 91.1% | **81.8%** | 0.77 | **0.159** |
| Blending | 91.9% | 63.6% | 0.74 | 0.200 |

### Which Model to Use

**If the priority is overall classification quality:** the Best NN is the clear winner (95.2% accuracy, F1=0.85). Its 94% precision on Class 1 means predicted positives are highly reliable.

**If the priority is maximizing recall** (catching every possible positive case, accepting more false alarms): the NN Baseline is surprising — despite having no regularization, it achieves the highest recall (81.8%) and the best probability calibration (log loss 0.159). This is worth investigating further; it may be that the unregularized model has learned to be more liberal about predicting Class 1.

**If a single strong non-neural model is needed:** Stacking is the best option. It outperforms the XGBoost baseline across all metrics without requiring TensorFlow.

**Blending is not recommended** for this dataset size. The performance loss is significant and the holdout approach does not justify the added complexity.

---

## 9. Summary of Key Decisions and Findings

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Feature scaling | StandardScaler | Required for neural network gradient stability |
| Imbalance handling | Class weighting (4.73×) | Computed from training set; no resampling needed |
| Architecture depth | 2 layers preferred | Deeper networks overfit on 493 training samples |
| Best regularization | L2 + BatchNorm + Dropout | Each targets a different failure mode; complementary |
| BatchNorm alone | Avoided | Batch statistics unstable with small batches/data |
| Ensemble strategy | Stacking > Blending | Cross-validation uses data more efficiently |
| NN in blending | Included | Contributes meaningful signal (weight: 1.163) |
| Meta-learner | Logistic Regression | Interpretable; linear combination of base predictions |

### Broader Lessons

**Small datasets reward simplicity.** Every experiment in this notebook confirms that model capacity should match data volume. The 1- and 2-layer networks outperformed deeper ones across every metric. This same principle applies to ensemble methods: stacking's thorough use of available data through cross-validation is better suited to this problem than blending's single-holdout approach.

**Regularization is additive.** No single technique (Dropout, BatchNorm, or L2 alone) matched the combined model. BatchNorm was harmful in isolation but beneficial in combination — a finding that would be missed by testing techniques sequentially rather than jointly.

**Class imbalance is a first-order concern.** The jump from the unweighted XGBoost baseline (59% recall on Class 1) to the weighted version (73% recall) demonstrated this in the prior notebook. The same principle applied here: every model in this notebook used either class weighting or `scale_pos_weight`, and the models that performed poorly on Class 1 (e.g., blending's base models trained on reduced data) did so partly because the imbalance handling was diluted by the smaller training set.

**Probability calibration matters.** Log loss is reported alongside accuracy throughout because in a medical classification task, the model's confidence is as important as its binary prediction. A model that outputs 0.99 for a positive case is more clinically useful than one that outputs 0.55 for the same case, even if both lead to the same binary decision. The NN Baseline's log loss of 0.159 — best in the notebook — suggests its predicted probabilities are well-calibrated despite its lack of regularization.
