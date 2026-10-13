# Kinyarwanda News Topic Classification (KINNEWS + AfroXLMR)

Individual NLP capstone, African Leadership University (Summative Project Sep26).
Task: **text classification for an African language**: assign a Kinyarwanda news article to one of 14 topics.

## Project links

| Resource | Link |
|---|---|
| GitHub repository | TODO: https://github.com/bianca255/kinnews-classifier |
| Demo video | TODO: add video URL |
| Fine-tuned model (Hugging Face) | https://huggingface.co/Bianca255/kinnews-afroxlmr |
| Live system | https://kin-news-topics-bianca255.streamlit.app |
| Report | TODO: add link or file name |

## Problem

Kinyarwanda is spoken by more than 10 million people but has few NLP resources. Automatic topic classification of news helps organise and search Kinyarwanda content. The model assigns an article to one of 14 topics: politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, relationship.

## Dataset

- **KINNEWS** (Niyongabo et al., 2020), cleaned version, loaded from the Hugging Face mirror `mteb/KinNewsClassification` (config `kinnews_cleaned`).
- Original train: 9,075 articles; test: 2,008 articles; 14 classes.
- The dataset has no validation split, so 10% of train was held out (stratified, seed 42): **8,167 train / 908 validation / 2,008 test**.
- Naturally occurring Rwandan news text; no synthetic data used.
- Mean article length is about 268 words (max 8,597).
- Checked: no exact duplicate texts in train, and no train/test text overlap.
- **Known issues:** the "cleaned" text is lowercased and punctuation and numbers are removed; some words are glued together; the classes are imbalanced (1,557 training articles for politics vs 92 for tourism); some test classes are tiny (tourism: 8 articles).

## Methodology

| Role | Model |
|---|---|
| Baseline | TF-IDF (1-2 grams, min_df=2, sublinear TF, 200k features) + Logistic Regression (balanced class weights) |
| Proposed | Fine-tuned **AfroXLMR-base** (`Davlan/afro-xlmr-base`) with a sequence-classification head |

Final model settings: learning rate 2e-5, batch size 8, 6 epochs, weight decay 0.01, warmup of 10% of steps, max length 512 tokens, fp16, seed 42. The best checkpoint is chosen by **validation macro-F1** (epoch 5). The test set is used only for reporting.

## Experiments

| # | Model | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|-------|-------------|---------------|---------------|
| 0 | TF-IDF + Logistic Regression (baseline) | 0.725 | 0.781 | 0.706 |
| 1 | AfroXLMR, 3 epochs, max_len 256 | 0.687 | 0.779 | 0.657 |
| 2 | AfroXLMR, 6 epochs, max_len 256 | 0.730 | 0.775 | 0.675 |
| 3 | AfroXLMR, 6 epochs, max_len 256, class-weighted loss | 0.722 | 0.780 | 0.689 |
| 4 | **AfroXLMR, 6 epochs, max_len 512 (final)** | **0.739** | **0.791** | 0.691 |

Experiments 1-3 used batch size 16 and experiment 4 used batch size 8. Full numbers are in `results/results.json`.

**Main findings**
- Training longer helped (experiment 1 was under-trained). Longer inputs (512 tokens) gave the best validation macro-F1 and test accuracy.
- Class-weighted loss gave no clear gain.
- The simple baseline is still competitive: it has the best test macro-F1, while AfroXLMR has the best accuracy. AfroXLMR is better on large classes (entertainment, relationship, politics) and worse on small ones (technology, culture).

## Error analysis (summary)

- The most frequent confusions are between the large classes, in both directions (politics to economy: 34, politics to health: 31, health to politics: 31, economy to politics: 27). This points to overlapping topics and label ambiguity in the dataset.
- The weakest classes are environment, history, education, culture and tourism (F1 about 0.41-0.62). Several have very few test articles, so the scores are noisy.
- Misclassified articles are not shorter than correctly classified ones (mean 277 vs 264 words).
- Full discussion is in the report.

## Repository structure

```
.
├── README.md
├── app.py                  # Gradio web app (loads the model from Hugging Face)
├── requirements.txt
├── notebooks/
│   └── kinnews_classification.ipynb   # data prep, baseline, fine-tuning, evaluation, error analysis
└── results/
    └── results.json        # numbers for every experiment
```

## How to reproduce

1. Open `notebooks/kinnews_classification.ipynb` in Google Colab and set the runtime to **T4 GPU**.
2. Run the cells in order: install packages, load the dataset, create the splits and save them to `data/`, run the TF-IDF baseline, then the fine-tuning experiments.
3. Install dependencies locally with `pip install -r requirements.txt` if running outside Colab.

## Run the web app

```bash
pip install -r requirements.txt
python app.py
```

This downloads the fine-tuned model from `Bianca255/kinnews-afroxlmr` (about 1 GB) and starts a local Gradio app. In Colab, change the last line of `app.py` to `demo.launch(share=True)` to get a temporary public link.

## Deployment

The model is hosted on the Hugging Face Hub and served through a Gradio interface. The free Hugging Face Gradio Space tier requires a PRO subscription, so the app is not permanently hosted. It is run from Colab with a temporary `gradio.live` link, which stops working when the session ends. The demo video shows the full workflow.

## Limitations

- The dataset text is lowercased and stripped of punctuation and numbers, so the app lowercases its input to match.
- Classes are imbalanced and some test classes are very small, so per-class scores are noisy.
- Labels appear to overlap (for example, politics vs health or history), which limits the achievable accuracy.
- The model always predicts one of 14 news topics, even for text outside the news domain.
- Results come from a single run per configuration (seed 42), so small differences between experiments may be within noise.

## Existing resources vs my own work

**Existing resources used:** the KINNEWS dataset and its Hugging Face mirror; the pretrained AfroXLMR model; the Hugging Face Transformers, Datasets, Hugging Face Hub, scikit-learn, PyTorch and Gradio libraries.

**My own work:** the train/validation split, data checks, the TF-IDF baseline, the fine-tuning setup and experiments (including the class-weighted loss and the sequence-length comparison), the evaluation and error analysis, and the Gradio app and deployment.

## References

- Niyongabo, R. A., Qu, H., Kreutzer, J., and Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi. *COLING 2020*.
- Alabi, J. O., Adelani, D. I., Mosbach, M., and Klakow, D. (2022). Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning. *COLING 2022*.
- Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *EMNLP 2020: System Demonstrations*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12.
- Abid, A. et al. (2019). Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild. *arXiv:1906.02569*.
