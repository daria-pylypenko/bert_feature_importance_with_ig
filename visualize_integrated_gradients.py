""" Visualize integrated gradients

for a certain model checkpoint, and a certain paragraph.

Input:
    input_data
    paragraph_id 
    output_image
    -model_dir
    -vocab_path
    --masked_input
    --pos_tags
    --detailed_pos_tags

    Some options have been used for earlier formats of input data and model.
    These are: -config_path, -model_path, --add_ne_tokens
    It is better to avoid using them, but to use model_dir and vocab_path instead,
    after saving both model and tokenizer with save_pretrained(), like:
    model.save_pretrained(model_dir), tokenizer.save_pretrained(tokenizer_dir)

Output:
    image_file (.html): table containing columns: true label,
       predicted_label, attribution_label, attribution_score,
       word importance (heatmap)
       Saved in the images/ folder

References: 
https://captum.ai/tutorials/Bert_SQUAD_Interpret
"""

import argparse
import math
import os
import pandas as pd
import simpletransformers
import torch
import transformers

from argparse import RawTextHelpFormatter
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from transformers import BertTokenizer, BertForSequenceClassification

cuda_device=0
device=torch.device("cuda:0")

def load_model(model_path="", add_ne_tokens=False, vocab_path="", config_path="", model_dir=""):
    """ Load BERT and the tokenizer from the given paths

    This function is messy, because we changed the input format many times,
    so it still works for several old ways of providing the input paths,
    in case I need to reproduce the paper results.
    But the preferred way of calling this function is:
        load_model(model_dir, vocab_path)
    where model_dir and vocab_path have been created with save_pretrained(), like:
    model.save_pretrained(model_dir) and model.save_pretrained(tokenizer_dir)

    Args:
        model_path (str)
        add_ne_tokens (bool)
        vocab_path (str)
        config_path (str)
        model_dir (str)

    Returns:
        model (torch.nn.DataParallel(transformers.BertForSequenceClassification))
        tokenizer (transformers.BertTokenizer)
    """

    if vocab_path:
        tokenizer = BertTokenizer.from_pretrained(vocab_path, max_len=512)
        # the tokenizer for detailed postags has do_lower_case set to True
        #print("tokenizer.do_lower_case: ", tokenizer.do_lower_case)
        # the tokenizer for regular postags also has do_lower_case set to True
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        # do_lower_case is True by default in this tokenizer

    #print(type(tokenizer.vocab)) # collections.OrderedDict
    #print(len(tokenizer.vocab.keys()))
    #input()

    if model_path:
        checkpoint = torch.load(model_path)
    #print(type(checkpoint))
    #print(checkpoint.keys())
    #print(type(checkpoint['model_state_dict']))
    # the checkpoints are type dict, with keys (['model_state_dict', 'optimizer_state_dict'])
    # model_state_dict is a collections.OrderedDict containing weights and biases

    if config_path:
        model = BertForSequenceClassification.from_pretrained(None,
                state_dict=checkpoint['model_state_dict'], config = config_path)
    elif model_dir:
        model = BertForSequenceClassification.from_pretrained(model_dir)
    else:
        model = BertForSequenceClassification.from_pretrained(
                               'bert-base-multilingual-uncased')
                               #num_labels = 2)  # not sure we need this

    if add_ne_tokens:
        #special_tokens = {'additional_special_tokens': ["[LOC]", "[MISC]", "[ORG]", "[PER]"]} 
        #tokenizer.add_special_tokens(special_tokens)
        # Previously we added the NE tokens as special tokens.
        # However, apparently special tokens do not get updated during fine-tuning.
        # Thus they were added as regular tokens (I think).
        # So we add them here as regular tokens too.
        # They are also modified so that they do not get split up during tokenization.
        # Previously though --add_ne_tokens flag needed to be used for both masked and unmasked data,
        # because apparently unmasked BERT was also trained with the vocabulary extended
        # by 4 more tokens (although obviously it has not seen them in fine-tuning).
        # TODO: Now I have to check if the code will still work with unmasked data
        # if I add the NE tokens as regular tokens, not as special tokens

        #print("BERT vocabulary size:", len(tokenizer))
        new_tokens = ["LOCC", "MISCC", "ORGG", "PERR"]
        #print(new_tokens)

        # check if the tokens are already in the vocabulary
        # we lowercase, since the tokens will get automatically lowercased when added to the vocabulary,
        # since our BERT is uncased
        #lowered_new_tokens = [token.lower() for token in new_tokens]
        #num_tokens_not_in_vocab = len(set(lowered_new_tokens) - set(tokenizer.vocab.keys()))
        #print(num_tokens_not_in_vocab, "new tokens not found in the vocabulary of the pretrained BERT")
        #for token in lowered_new_tokens:
        #    print(token, token in tokenizer.vocab)

        #print(type(tokenizer.vocab)) # collections.OrderedDict
        #print(len(tokenizer.vocab.keys())) # 105879

        # Print some vocabulary items
        # Format "token" : id
        #start = 50000
        #end = start + 10
        #i = 0
        #for key, value in tokenizer.vocab.items():
        #    if i >= start:
        #        print(key, value)
        #    i += 1
        #    if i == end:
        #        break
        #input()

        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer))
        #print("BERT vocabulary size:", len(tokenizer))

        # tokenizer.vocab() will only return the base vocabulary, without added tokens
        # tokenizer.get_vocab() should return everything, but needs testing
        #print(tokenizer.get_added_vocab())
        # The tokens get automatically lowercased when we add them to the vocabulary,
        # since our BERT is uncased

    if (not config_path) and model_path: # if we have config_path, we have loaded already the state dict into the model
                                         # if we don't have model_path, we have not loaded the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded state dict")

    model.to(device)
    # DataParallel does not change anything currently,
    # but could become useful if I make the data be processed in batches
    model = torch.nn.DataParallel(model)
    return model, tokenizer


def custom_forward(input_tensor):
     """Custom forward that selects only logits from the output
        of the actual forward"""
     model_output = model.module(input_tensor)
     return model_output.logits


def summarize_attributions(attributions):
    """Find a mean attribution value for each embedding"""
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def generate_tensors_for_ig(paragraph, tokenizer, max_len=math.inf):
    """Generate a paragraph tensor and a reference tensor

    Args:
        paragraph (str): stripped paragraph in the original case
        tokenizer (transformers.BertTokenizer)
        max_len (int): maximal number of tokens; if it is surpassed,
                  the sequence will be cut to meet the required length

    Returns:
        input_tensor (torch.LongTensor)
        ref_tensor (torch.LongTensor): pad tokens of the same shape
             as the original paragraph
    """
    input_ids = tokenizer.encode(paragraph)
    input_len = len(input_ids)
    if input_len > max_len:
        input_len = max_len
        input_ids = input_ids[:(max_len-1)] + [tokenizer.sep_token_id] 
    ref_input_ids = [tokenizer.cls_token_id] +\
                 [tokenizer.pad_token_id]*(input_len-2) +\
                      [tokenizer.sep_token_id]


    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    ref_tensor = torch.tensor(ref_input_ids, dtype=torch.long).unsqueeze(0)

    input_tensor = input_tensor.to(device)
    ref_tensor = ref_tensor.to(device)
    return input_tensor, ref_tensor


def tokenize(paragraph, tokenizer, max_len=math.inf):
    """ Tokenize the paragraph

    Args:
        paragraph (str): stripped paragraph in the original case
        tokenizer (transformers.BertTokenizer): tokenizer
        max_len (int): maximal number of tokens; if it is surpassed,
                  the sequence will be cut to meet the required length

    Returns:
        tokens (list of str)?
    """
    #TODO: we encode twice, at first for tensor, then for the tokens. Does this take a lot of time?
    input_ids = tokenizer.encode(paragraph)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    if len(tokens) > max_len:
        tokens = tokens[:(max_len-1)] + [tokenizer.sep_token]
    return tokens


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_data", help=".csv format")
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
    parser.add_argument("paragraph_id", type=int, help="Line number, starting from 0")
    parser.add_argument("output_image", help="Output image filename")
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

    # Deprecated arguments:
    parser.add_argument("-config_path", help="Deprecated: A path to a config file")
    parser.add_argument("-model_path", help="Deprecated: Model file with .pth extension")
    parser.add_argument("--add_ne_tokens", action="store_true", 
                        help="Deprecated: Add Named Entity tokens: [LOC], [MISC], [ORG], [PER]")
    args = parser.parse_args()

    # Load the model and initialize the tokenizer
    print("Loading the model...")
    model, tokenizer = load_model(model_path=args.model_path, add_ne_tokens=args.add_ne_tokens,
                                     vocab_path=args.vocab_path, config_path=args.config_path,
                                     model_dir=args.model_dir)
    print("Done.")

    # .module() because we use DataParallel
    model.module.eval()
    model.module.zero_grad()

    # Load the paragraph and its ground truth label
    frame = pd.read_csv(args.input_data)
    if args.masked_input:
        paragraph = frame['masked'].iloc[args.paragraph_id]

    elif args.pos_tags:
        paragraph = frame['pos'].iloc[args.paragraph_id]

    elif args.detailed_pos_tags:
        paragraph = frame['tag'].iloc[args.paragraph_id]

    else:
        paragraph = frame['text'].iloc[args.paragraph_id]
    paragraph = paragraph.strip()

    gt_class = int(frame['label'].iloc[args.paragraph_id])


    # Convert the paragraph to an input tensor,
    # and create a reference tensor for IG
    input_tensor, ref_tensor = generate_tensors_for_ig(paragraph, tokenizer)


    print("Computing the attributions...")
    # Compute the attributions to the embeddings, and find a mean attribution
    # for each embedding
    # We compute attribution of the correct class
    lig = LayerIntegratedGradients(custom_forward, model.module.bert.embeddings)
    attributions, delta = lig.attribute(inputs=input_tensor, baselines=ref_tensor,
                                      target=gt_class, return_convergence_delta=True)
    attributions_sum = summarize_attributions(attributions)
    print("Done.")

    # Prepare data for visualization
    with torch.no_grad():
         logits = custom_forward(input_tensor)
         print("logits", logits)
         
    pred_prob = torch.max(torch.softmax(logits[0], dim=0))
    print("pred_prob", pred_prob)
    pred_class = torch.argmax(logits)
    print("pred_class", pred_class)
    attr_class = gt_class # Class for which computed attribution
    print("gt_class", gt_class)
    print("attr_class", attr_class)
    attr_score = attributions_sum.sum() # Summed for the whole paragraph 
    tokens = tokenize(paragraph, tokenizer)

    # Visualize
    visualization = viz.VisualizationDataRecord(attributions_sum,
	                     pred_prob, pred_class, gt_class, attr_class,
                             attr_score, tokens, delta)                                      


    image = viz.visualize_text([visualization])

    # Save as an .html file
    image_dir = "images"
    f_image_name = os.path.join(image_dir, args.output_image + ".html")
    with open(f_image_name, "w") as f_image:
        f_image.write(image.data)








