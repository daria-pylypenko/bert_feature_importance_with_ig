# BERT feature importance using Integrated Gradients

Code used for Integrated Gradients experiments in the publication:
> Angana Borah, Daria Pylypenko, Cristina España-Bonet, and Josef van Genabith. 2023. Measuring spurious correlation in classification: “Clever Hans” in translationese. In Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing, pages 196–206, Varna, Bulgaria. INCOMA Ltd., Shoumen, Bulgaria.

Scripts for computing and visualizing BERT feature importance
Attribution method: Integrated Gradients (IG)

### Environment
Installing the conda environment that was used for the IG experiments in the publication:
```bash
conda env create -f environment_ig_for_bert.yml
```

### Input arguments
Model and tokenizer have to be both saved with *save\_pretrained()*, like:
*model.save\_pretrained(model\_dir)* and *tokenizer.save\_pretrained(tokenizer\_dir)*

Data must be in the .csv format and have the following fields:
1. either "text", or "masked", or "pos", or "tag"
2. "label"

### find\_tokens\_with\_highest\_attribution.py
Find top-N tokens with highest average IG attribution score in the dataset, for each of the 2 classes: originals and translationese.
Given the dataset, for each paragraph compute the IG scores of all tokens, with respect to the ground truth class.
For each class we find average IG score per token across the dataset.
Then find top N tokens for each class.

#### Example commands:

Unmasked data
```bash
python find_tokens_with_highest_attribution.py dataset-name.csv top_attributions__model-name__dataset-name -model_dir model_dir/ -vocab_path tokenizer_dir/
```

Masked data
```bash
python find_tokens_with_highest_attribution.py dataset-name.csv top_attributions__model-name__dataset-name -model_dir model_dir/ -vocab_path tokenizer_dir/ --masked_input
```

Detailed POS tags (TIGER Treebank tags)
```bash
python find_tokens_with_highest_attribution.py dataset-name.csv top_attributions__model-name__dataset-name -model_dir model_dir/ -vocab_path tokenizer_dir/ --detailed_pos_tags
```

Regular POS tags (UPOS)
```bash
python find_tokens_with_highest_attribution.py dataset-name.csv top_attributions__model-name__dataset-name -model_dir model_dir/ -vocab_path tokenizer_dir/ --pos_tags
```

Adding ```--compute_accuracy``` flag to also compute classification accuracy on the dataset (example for regular POS tags):
```bash
python find_tokens_with_highest_attribution.py dataset-name.csv top_attributions__model-name__dataset-name -model_dir model_dir/ -vocab_path tokenizer_dir/ --pos_tags --compute_accuracy
```


### visualize\_integrated\_gradients.py
Visualize IG token attributions for a certain model, and a certain paragraph.

#### Example command:

Masked data, visualize paragraph 0
```bash
python visualize_integrated_gradients.py dataset-name.csv 0 image__model-name__dataset-name__0 -model_dir model_dir/ -vocab_path tokenizer_dir/ --masked_input
```

