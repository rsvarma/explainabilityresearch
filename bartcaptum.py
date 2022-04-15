from wsgiref import validate
from transformers import BartTokenizer, BartForConditionalGeneration, utils
import pdb
import torch
from captum.attr import LayerConductance, LayerIntegratedGradients,IntegratedGradients
import modelutils

def bart_forward_func(attr_embeds,other_input_embeds,bart,pred_idx,index,attr_mode="enc"):
    if attr_mode=="enc":
        outputs = bart(inputs_embeds=attr_embeds,decoder_inputs_embeds=other_input_embeds)
    elif attr_mode=="dec":
        outputs = bart(inputs_embeds=other_input_embeds,decoder_inputs_embeds=attr_embeds)
    pred = outputs.logits[:,index-1,pred_idx]
    return pred

def generate_ref_sequences(input_ids,embed,tokenizer):
    ref_input_ids = torch.zeros_like(input_ids)
    ref_input_ids[0,0] = tokenizer.bos_token_id
    ref_input_ids[0,1:-1] = tokenizer.pad_token_id
    ref_input_ids[0,-1] = tokenizer.eos_token_id
    ref_input_embeds = embed(ref_input_ids)
    return ref_input_embeds


# Calculates value of sum of path integral for verification purposes by subtracting forward function output on baseline from input
# Based on Proposition 1 from https://arxiv.org/pdf/1703.01365.pdf
# Inputs: 
# input_forward_func_args: Tuple; Tuple containing function arguments for model's forward function that will produce output on input
# baseline_forward_func_args: Tuple; Tuple containing function arguments for model's forward function that will produce output on baseline
#
# Returns:
# validation_diff: int; Integer representing the difference described in Proposition 1 discussed above
def validate_attribution_sum(input_forward_func_args,baseline_forward_func_args,forward_func):
    input_pred = forward_func(*input_forward_func_args)
    baseline_pred = forward_func(*baseline_forward_func_args)
    validation_diff = input_pred-baseline_pred
    return validation_diff.item()


def main():    

    device = torch.device("cuda")
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    bart = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)


    encoder_input_ids = tokenizer("The House Budget Committee voted Saturday to pass a $3.5 trillion spending bill", return_tensors="pt", add_special_tokens=True).input_ids
    decoder_input_ids = tokenizer("The House Budget Committee passed a spending bill.", return_tensors="pt", add_special_tokens=True).input_ids

    analyze_idx = 5

    embed = torch.nn.Embedding(bart.model.shared.num_embeddings,bart.model.shared.embedding_dim,bart.model.shared.padding_idx)
    embed.weight.data = bart.model.shared.weight.data

    ref_encoder_embeds = generate_ref_sequences(encoder_input_ids,embed,tokenizer)
    ref_decoder_embeds = generate_ref_sequences(decoder_input_ids[:,:analyze_idx],embed,tokenizer)

    encoder_embeds = embed(encoder_input_ids)
    decoder_embeds = embed(decoder_input_ids[:,:analyze_idx])

    decoder_text = modelutils.replace_special_bart_tokens(modelutils.get_id_text(decoder_input_ids,tokenizer)) 
    decoder_text = decoder_text[:analyze_idx]

    encoder_text = modelutils.replace_special_bart_tokens(modelutils.get_id_text(encoder_input_ids,tokenizer)) 

    pred_idx = decoder_input_ids[0,analyze_idx]

    bart.to(device)
    ig = IntegratedGradients(bart_forward_func)

    encoder_input_attributions = ig.attribute(inputs=encoder_embeds.to(device),baselines = ref_encoder_embeds.to(device),additional_forward_args= (decoder_embeds.to(device),bart,pred_idx,analyze_idx),n_steps=1000,internal_batch_size=300)
    val_diff = validate_attribution_sum((encoder_embeds.to(device),decoder_embeds.to(device),bart,pred_idx,analyze_idx),(ref_encoder_embeds.to(device),decoder_embeds.to(device),bart,pred_idx,analyze_idx),bart_forward_func)
    print(f"following should be close to equal {val_diff} == {encoder_input_attributions.sum().item()}")
    print("\nEncoder Input Attributions")
    encoder_input_attributions = encoder_input_attributions.sum(dim=-1)
    encoder_input_attributions = encoder_input_attributions/torch.norm(encoder_input_attributions)    
    concat_attr_text_score =  [ f"{i}: {j:.4f}\n" for i,j in zip(encoder_text,encoder_input_attributions[0].tolist())]
    print("".join(concat_attr_text_score))

    decoder_attributions = ig.attribute(inputs=decoder_embeds.to(device),baselines=ref_decoder_embeds.to(device),additional_forward_args=(encoder_embeds.to(device),bart,pred_idx,analyze_idx,"dec"),n_steps=1000,internal_batch_size=300)
    val_diff = validate_attribution_sum((decoder_embeds.to(device),encoder_embeds.to(device),bart,pred_idx,analyze_idx,"dec"),(ref_decoder_embeds.to(device),encoder_embeds.to(device),bart,pred_idx,analyze_idx,"dec"),bart_forward_func)
    print(f"following should be close to equal {val_diff} == {decoder_attributions.sum().item()}")
    print("\nDecoder Attributions")
    decoder_attributions = decoder_attributions.sum(dim=-1)
    decoder_attributions = decoder_attributions/torch.norm(decoder_attributions)
    concat_attr_text_score =  [ i+": ,"+str(j) for i,j in zip(decoder_text,decoder_attributions[0].tolist())]
    print("".join(concat_attr_text_score))    


if __name__ == "__main__":
    main()
