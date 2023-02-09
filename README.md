# Cross-lingual Approaches for the Detection of Adverse Drug Reactions in German from a Patient’s Perspective (LREC 2022)

To get access to data and/or models, please contact [Lisa](lisa.raithel@dfki.de).

If you use this data, please cite our paper:

Raithel, L., Thomas, P., Roller, R., Sapina, O., Möller S., Zweigenbaum, P. (2022), [Cross-lingual Approaches for the Detection of Adverse Drug Reactions in German from a Patient’s Perspective](https://aclanthology.org/2022.lrec-1.388). In Proceedings of the 13th Language Resources and Evaluation Conference (LREC), Marseille, June 2022


```
@inproceedings{raithel-etal-2022-cross,
    title = "Cross-lingual Approaches for the Detection of Adverse Drug Reactions in {G}erman from a Patient{'}s Perspective",
    author = {Raithel, Lisa  and
      Thomas, Philippe  and
      Roller, Roland  and
      Sapina, Oliver  and
      M{\"o}ller, Sebastian  and
      Zweigenbaum, Pierre},
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.388",
    pages = "3637--3649",
    abstract = "In this work, we present the first corpus for German Adverse Drug Reaction (ADR) detection in patient-generated content. The data consists of 4,169 binary annotated documents from a German patient forum, where users talk about health issues and get advice from medical doctors. As is common in social media data in this domain, the class labels of the corpus are very imbalanced. This and a high topic imbalance make it a very challenging dataset, since often, the same symptom can have several causes and is not always related to a medication intake. We aim to encourage further multi-lingual efforts in the domain of ADR detection and provide preliminary experiments for binary classification using different methods of zero- and few-shot learning based on a multi-lingual model. When fine-tuning XLM-RoBERTa first on English patient forum data and then on the new German data, we achieve an F1-score of 37.52 for the positive class. We make the dataset and models publicly available for the community.",
}
```
