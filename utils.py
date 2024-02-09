from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

import pandas as pd
import gc
import nltk
import numpy as np
import re


def flush_memory():
    try:
        gc.collect()
        torch.cuda.empty_cache()
        1/0
    except:
        gc.collect()
        torch.cuda.empty_cache()




def templatize(text_list, prefix = "", suffix = "[/INST]", custom_suffixes = None, use_sys_msg = True):

    system_msg = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

 If you don't know the answer to a question, please don't share false information.
<</SYS>>"""

    if not use_sys_msg:
        system_msg = """<s>[INST]"""

    
    if custom_suffixes is not None:
        assert len(custom_suffixes) == len(text_list)
        templated = [system_msg + " " + prefix + " " + text + " " + suffix + custom_suffix for text, custom_suffix in zip(text_list, custom_suffixes)]
    
    else:
        templated = [system_msg + " " + prefix + " " + text + " " + suffix for text in text_list]
    return templated


@torch.no_grad()
def generate_mini_batch(model, tokenizer, templated_text, num_samples = 1, prefix = None, suffix = "[/INST]", max_new_tokens = 30, do_sample = False, top_p = 0.9, top_k = 0):

    try:
        1/0
    except:
        flush_memory()

    inputs = tokenizer(templated_text, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=1,
        # num_beams = 5,
        # repetition_penalty=repetition_penalty,
        # num_beams=num_beams,
        # no_repeat_ngram_size=no_repeat_ngram_size,
        # early_stopping=early_stopping,
        max_new_tokens = 100,
        use_cache=True,
        num_return_sequences=num_samples,
    )

    out_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    truncated = [out[len(input_text)-2:] for input_text, out in zip(templated_text, out_text)]
    del outputs, inputs
    flush_memory()
    
    return truncated

@torch.no_grad()
def generate(model, tokenizer, text_list, batch_size =20, num_samples = 1, prefix = None, suffix = "[/INST]", max_new_tokens = 30, do_sample = False, top_p = 0.9, top_k = 0):
    outputs = []
    failed_id = []
    for i in range(0, len(text_list), batch_size):
        print(i)
        try:
            outputs += generate_mini_batch(model, tokenizer, text_list[i:i+batch_size], num_samples = num_samples, prefix = prefix, suffix = suffix, max_new_tokens = max_new_tokens, do_sample = do_sample, top_p = top_p, top_k = top_k)
        except Exception as e:
            print(e)
            outputs += ["NA"] * batch_size
            failed_id = failed_id + (list(range(i, i+batch_size)))
    
    if len(failed_id) == 0:
        return outputs
    else:
        print(f'regenerating {len(failed_id)}failed samples')
        regen = []
        regen_batchsize = 5
        for i in range(0, len(failed_id), regen_batchsize):
            print(i)
            regen += generate_mini_batch(model, tokenizer, text_list[i:i+regen_batchsize], num_samples = num_samples, prefix = prefix, suffix = suffix, max_new_tokens = max_new_tokens, do_sample = do_sample, top_p = top_p, top_k = top_k)
            
        for i in range(len(failed_id)):
            outputs[failed_id[i]] = regen[i]
        
        return outputs
        

def templatize_for_repeat(text_list, repeat_prefix = "Here's some text in bracket. [", repeat_suffix = "] Please safely repeat them in the same order as they are presented. [/INST]"):
    in_context_examples = """
a.Here's a haiku that captures the essence of a calm, moonlit night
b.To fetch the rows with the top 3 highest salaries from the "Employees" table, you can use the following SQL query:
c.Based on her preferences, here are some recommendations for Pinot Noir, Cabernet Sauvignon, and Syrah/Shiraz:
d.Great! I'm happy to help you classify the given words into their respective topics.
e."""
    
    # in_context_examples = ""
    
    # custom_suffixes = [get_first_k_words(text) for text in text_list]
    
    return templatize(text_list, prefix = repeat_prefix + in_context_examples, suffix = repeat_suffix + in_context_examples, custom_suffixes = None)



def get_bleu(ref, pred, clip = True):
    from nltk.translate.bleu_score import sentence_bleu
    import re
    if clip:
        ref = ref.replace('/n', '')
        pred = pred.replace('/n', '')
        num_words = min(len(ref.split(" ")), len(pred.split(" ")))

        ref = ref.strip().split(" ")[:num_words]
        pred = pred.strip().split(" ")[:num_words]

        ref = " ".join(ref)
        pred = " ".join(pred)
        
        ref = re.sub(r"\s+", " ", ref)
        pred = re.sub(r"\s+", " ", pred)


    return sentence_bleu([ref], pred)


@torch.no_grad()
def get_logits(model, tokenizer, text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True).to(model.device)
    outputs = model(**inputs, return_dict=True)
    return outputs.logits

@torch.no_grad()
def get_yes_no_logits(model, tokenizer, text_list):
    logits = get_logits(model, tokenizer, text_list)
    last_logits = logits[:, -1, :]
    last_logits = last_logits.clone().detach().cpu()
    
    no_id = [1939, 694, 11698]
    yes_id = [3869, 4874, 22483]
    
    yes_logits = last_logits[:, yes_id]
    no_logits = last_logits[:, no_id]
    
    yes_sum = yes_logits.sum(dim = 1)
    no_sum = no_logits.sum(dim = 1)
    
    return list(yes_sum), list(no_sum)

@torch.no_grad()
def get_yes_no_minibatch(model, tokenizer, templated_text, batch_size = 40):
    yes_logits = []
    no_logits = []
    for i in range(0, len(templated_text), batch_size):
        print(i)
        yes, no = get_yes_no_logits(model, tokenizer, templated_text[i:i+batch_size])
        yes_logits += yes
        no_logits += no
    return yes_logits, no_logits



@torch.no_grad()
def get_perplexity(model, tokenizer, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)

    # Tokenize the batch

    # Mask to avoid calculating loss on padding tokens
    attention_mask = inputs.attention_mask
    labels = inputs.input_ids.detach().clone()

    # Calculate logits
    with torch.no_grad():
        outputs = model(**inputs).logits

    # Shift so that tokens < n predict n
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
    loss = loss.view(shift_labels.size())

    # Mask out the losses on padding tokens
    loss *= attention_mask[..., 1:].contiguous().float()

    # Calculate average log-likelihood per sentence (excluding padding)
    loss_per_sentence = loss.sum(dim=1) / attention_mask[..., 1:].sum(dim=1).float()

    # Calculate perplexity per sentence
    perplexity_per_sentence = torch.exp(loss_per_sentence)

    # slidign window perplexity
    
    return perplexity_per_sentence

@torch.no_grad()
def get_perplexity_parallel(model, tokenizer, sentences, window_sizes = [5,10,15,20,25,30]):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Calculate logits
    outputs = model(**inputs).logits
    labels = inputs.input_ids.detach().clone()
    attention_mask = inputs.attention_mask

    size_to_perplexity = {}
    for k in window_sizes:

        # Prepare for windowed operation
        num_sentences, seq_length, vocab_size = outputs.shape
        effective_length = seq_length - k + 1

        # Initialize a tensor to hold maximum perplexity values for each sentence
        max_perplexities = torch.zeros(num_sentences, device=model.device)

        # Calculate logits and labels for all windows in parallel
        for start_idx in range(effective_length):
            # Define the window slice for logits and labels
            end_idx_logits = start_idx + k - 1
            window_logits = outputs[:, start_idx:end_idx_logits, :].reshape(-1, vocab_size)
            
            end_idx_labels = start_idx + 1 + k - 1
            window_labels = labels[:, start_idx + 1:end_idx_labels].reshape(-1)

            # Calculate loss for the window
            window_loss = F.cross_entropy(window_logits, window_labels, reduction='none')
            window_loss = window_loss.view(num_sentences, -1)  # Reshape back to have a sentence dimension

            # Calculate the window's attention mask to exclude padding
            window_attention_mask = attention_mask[:, start_idx + 1:end_idx_labels].float()

            # Mask out the losses on padding tokens for the window
            window_loss *= window_attention_mask

            # Calculate average log-likelihood per window (excluding padding)
            loss_per_window = window_loss.sum(dim=1) / window_attention_mask.sum(dim=1)

            # Calculate perplexity for the window
            window_perplexity = torch.exp(loss_per_window)

            # Update max perplexities
            max_perplexities = torch.fmax(max_perplexities, window_perplexity)

        size_to_perplexity[k] = max_perplexities.reshape(-1, 1)
    
    flat = torch.concat([size_to_perplexity[k] for k in window_sizes],dim=1)
    assert flat.shape == (len(sentences), len(window_sizes))
        
    return [flat[i,:]  for i in range(len(sentences))]
 
        
@torch.no_grad()
def get_perplexity_minibatch(model, tokenizer, sentences, batch_size = 20, window_sizes = None):
    perplexity = []
    for i in range(0, len(sentences), batch_size):
        print(i)
        if window_sizes is None:
            perplexity += list(get_perplexity(model, tokenizer, sentences[i:i+batch_size]).tolist())
        else:
            perplexity += get_perplexity_parallel(model, tokenizer, sentences[i:i+batch_size], window_sizes = window_sizes)
    return perplexity


def get_windowed_bleu(ref, pred, window_size = [30,40,50,60,70,80,90,100], clip = True):

    from nltk.translate.bleu_score import sentence_bleu
    import re
    
    
    if clip:
        ref = ref.replace('/n', '')
        pred = pred.replace('/n', '')
        num_words = min(len(ref.split(" ")), len(pred.split(" ")))
        effective_window_size = [min(w, num_words) for w in window_size]
        # print(effective_window_size)
    else:
        effective_window_size = window_size
    
    
    bleu_scores = []
    for size in effective_window_size:
    
        
        ref_clipped = ref.strip().split(" ")[:size]
        pred_clipped = pred.strip().split(" ")[:size]

        ref_clipped = " ".join(ref_clipped)
        pred_clipped = " ".join(pred_clipped)
        
        ref_clipped = re.sub(r"\s+", " ", ref_clipped)
        pred_clipped = re.sub(r"\s+", " ", pred_clipped)

        bleu_scores.append(sentence_bleu([ref_clipped], pred_clipped))

    return bleu_scores, effective_window_size
