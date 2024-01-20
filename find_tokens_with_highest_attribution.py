"""Find the most attributed tokens (on average) in the dataset

Find top-N tokens with highest average Integrated Gradients (IG) attribution score in the dataset, for each of the 2 classes: originals and translationese.
Given the dataset, for each paragraph compute the IG scores of all tokens, with respect to the ground truth class.
For each class we find average IG score per token across the dataset.
Then find top N tokens for each class.

Input:
    input_data
    output_file_id
    -model_dir
    -vocab_path
    -N
    --masked_input
    --pos_tags
    --detailed_pos_tags
    --compute_accuracy

    Some options have been used for earlier formats of input data and model.
    These are: -config_path, -model_path, --add_ne_tokens
    It is better to avoid using them, but to use model_dir and vocab_path instead,
    after saving both model and tokenizer with save_pretrained(), like:
    model.save_pretrained(model_dir), tokenizer.save_pretrained(tokenizer_dir)
      
Output:
    2 files: output_file_id_org.txt and output_file_id_trans.txt
       (will be written into the folder highest_attribution_tokens/)
       top-N tokens with highest average activation for each class

    If --compute_accuracy flag is given,
    the file with accuracy will be written in the accuracies/ folder,
    and will be named {output_file_id}_acc.txt

References:
https://captum.ai/tutorials/Bert_SQUAD_Interpret
"""


import argparse
import logging as log
import os
import pandas as pd
import spacy
import torch

from argparse import RawTextHelpFormatter
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from collections import defaultdict
from tqdm import tqdm
from visualize_integrated_gradients import generate_tensors_for_ig, tokenize,\
            summarize_attributions, load_model


gpu_device = torch.device("cuda:0")
cpu_device = torch.device("cpu")


def custom_forward(input_tensor):
     """Custom forward that selects only logits from the output
        of the actual forward"""
     model_output = model.module(input_tensor)
     return model_output.logits



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_data", help=".csv format, must contain fields: "
                                               "text/masked/pos/tag, label")
    # Fields: ,iid,src,native_speaker,original,dest,text,direct,label,masked
    # iid - id
    # src - original language of the paragraph
    # native_speaker - was produced by a native speaker of the original language (yes/no)
    # original - paragraph in the original language
    # dest - language of the paragraph
    #        (the same as the src for originals,
    #         different for translationese)
    # text - paragraph in "dest" language, not lowercased, not tokenized, not NE-masked
    # direct - translation was made directly, without a pivot language (yes/no)
    # label - 0 for originals, 1 for translationese
    # masked - not tokenized, not lowercased, only NE-masked

    # For the pos-tagged data it is iid, src, native_speaker, original, dest, text, direct, label, tag, pos
    # tag - the detailed pos tags
    # pos - regular pos tags
    parser.add_argument("output_file_id", help="file id for writing the results")


    parser.add_argument("-N", type=int, default=100, help="Top-N tokens will be written "
                                                                  "in the output files")
    parser.add_argument("-vocab_path", help="A path to a tokenizer folder")
    parser.add_argument("-model_dir", help="Model directory")

    parser.add_argument("--masked_input", action="store_true",
                       help="take the NE-masked input (\"masked\" field)")
    parser.add_argument("--pos_tags", action="store_true",
                        help="take the regular pos tags (the field \"pos\")")
    parser.add_argument("--detailed_pos_tags", action="store_true",
                        help="take the detailed pos tags (the field \"tag\")")

    # If neither masked_input, pos_tags or detailed_pos_tags flag is given,
    # takes unmasked input ("text" field)

    # I should consider making [masked_input, pos_tags, detailed_pos_tags]
    # into mutually exclusive options of the same argument

    parser.add_argument("--compute_accuracy", action="store_true",
            help="Compute the classification accuracy and write into file")

    # Deprecated arguments (technically they still work, and I keep them,
    # because I'll need them if I want to reproduce the experiments for the paper
    # with the exact same models and data that we used.)
    parser.add_argument("-config_path", help="Deprecated: A path to a config file")
    parser.add_argument("-model_path", help="Deprecated: Model file with .pth extension")
    parser.add_argument("--add_ne_tokens", action="store_true",
                        help="Deprecated: Add Named Entity tokens: [LOC], [MISC], [ORG], [PER]")
    
    args = parser.parse_args()


    # Load the model
    print("Loading the model...")
    model, tokenizer = load_model(model_path=args.model_path, add_ne_tokens=args.add_ne_tokens,
                                     vocab_path=args.vocab_path, config_path=args.config_path,
                                     model_dir=args.model_dir)
    print("Done.")
    #input()


    model.module.eval()
    model.module.zero_grad()
    # I don't think we have to do this for every paragraph, but I am not sure
    # Here they explain that it is not necessary: https://github.com/pytorch/captum/issues/590


    # Read in the data
    frame = pd.read_csv(args.input_data)



    org_token_attributions = defaultdict(float)
    org_token_counts = defaultdict(int)
    trans_token_attributions = defaultdict(float)
    trans_token_counts = defaultdict(int)

    lig = LayerIntegratedGradients(custom_forward, model.module.bert.embeddings)


    if args.compute_accuracy:
        accuracy = 0

    #TODO: try batches
    for i in tqdm(range(len(frame))):

        if args.masked_input:
            paragraph = frame['masked'].iloc[i]

        elif args.pos_tags:
            paragraph = frame['pos'].iloc[i]

        elif args.detailed_pos_tags:
            paragraph = frame['tag'].iloc[i]


        else:
            paragraph = frame['text'].iloc[i]
        paragraph = paragraph.strip()
        #print(paragraph)
        #input()

        gt_class = int(frame['label'].iloc[i])

        # Generate the paragraph and the reference tensor
        # I empirically verified that longer sequences do not fit on GPUs
        # Actually, up till around 245 works, but I wanted to make a slightly lower boundary 
        input_tensor, ref_tensor = generate_tensors_for_ig(paragraph, tokenizer, max_len=240)
        #print("Input tensor")
        #print(input_tensor)
        #print("input tensor shape")
        #print(input_tensor.shape)
        #input()

        tokens = tokenize(paragraph, tokenizer, max_len=240)
        #print("Tokens")
        #print(tokens)
        #print("len(tokens)")
        #print(len(tokens))
        #input()

        # Compute attributions
        attributions = lig.attribute(inputs=input_tensor, baselines=ref_tensor,
                                      target=gt_class)
        attributions = summarize_attributions(attributions)
        attributions = attributions.cpu().numpy()
        #print("Attributions")
        #print(attributions)
        #print("Attributions shape")
        #print(attributions.shape)
        #input()


        # Sum the attributions and the counts
        #assert len(tokens) == len(attributions)
        if gt_class==0:
            for token, attr in zip(tokens, attributions):
                org_token_attributions[token] += attr
                org_token_counts[token] += 1
        elif gt_class==1:
            for token, attr in zip(tokens, attributions):
                trans_token_attributions[token] += attr
                trans_token_counts[token] += 1

        # Compute accuracy as a sanity check
        if args.compute_accuracy:
             with torch.no_grad():
                 logits = custom_forward(input_tensor)
                 #print("logits", logits)
                 #input()

             pred_class = torch.argmax(logits)
             #print("pred_class", pred_class)
             #input()

             if gt_class == pred_class:
                 accuracy += 1

    # Compute mean attributions for the original class
    for token in org_token_attributions:
        org_token_attributions[token] = org_token_attributions[token] / org_token_counts[token]

    # Compute mean attributions for the translationese class
    for token in trans_token_attributions:
        trans_token_attributions[token] = trans_token_attributions[token] / trans_token_counts[token]


    # Sort 
    org_token_attributions = [(k, v) for k, v in sorted(org_token_attributions.items(),
                                           key=lambda item: item[1], reverse=True)]
    trans_token_attributions = [(k, v) for k, v in sorted(trans_token_attributions.items(),
                                           key=lambda item: item[1], reverse=True)]

    # Write into file
    output_folder = "highest_attribution_tokens"

    f_org_name = os.path.join(output_folder, args.output_file_id + "_org.txt")
    with open(f_org_name, "w") as f_org:
        for i, (token, attr) in enumerate(org_token_attributions[:args.N]):
            f_org.write(str(i) + " " + token + ": " + str(attr) + "\n")

    f_trans_name = os.path.join(output_folder, args.output_file_id + "_trans.txt")
    with open(f_trans_name, "w") as f_trans:
        for i, (token, attr) in enumerate(trans_token_attributions[:args.N]):
            f_trans.write(str(i) + " " + token + ": " + str(attr) + "\n")


    # Normalize accuracy and write into file
    if args.compute_accuracy:
        accuracy = (accuracy / len(frame)) * 100
        f_acc_name = os.path.join("accuracies", args.output_file_id + "_acc.txt")
        with open(f_acc_name, "w") as f_acc:
            f_acc.write(str(accuracy) + "\n")
