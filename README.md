# Code of KDD2024 paper: Dataset Regeneration for Sequential Recommendation

## Procedure to reproduce

- Download datasets
  - [Amazon](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/)
  - [Yelp](https://github.com/salesforce/ICLRec)
- Preprocess dataset with scirpts in `dataset/`
  - Amazon: `dataset/preprocess_single.ipynb`
  - Yelp: `dataset/preprocess_yelp-small.ipynb`
- Regenerating dataset with DR4SR
  - Construct the pre-training task
    - Generate rule-based pattern with `pattern_generator_seq2seq.ipynb`
    - Build connection between the original sequences and the patterns with `find_relation.ipynb`
  - Pre-train the regenerator with `translation_condition2.ipynb`
  - Regenerate dataset segments by the hybrid inference in `inference_con2.py`
  - Merge dataset in `generate_data.ipynb`
- Run target models based on the regenerated dataset
  - `python run.py -m BACKBONE -d DATASET`
  - 'BACKBONE' includes [GRU4Rec, SASRec, FMLP, GNN, CL4SRec]
- Run target models based on the personalized version of the regenerated dataset
  - Change 'sub_model' option to one of the target models in `configs/metamodel7.yaml`
  - `python run.py -m MetaModel7 -d DATASET`

