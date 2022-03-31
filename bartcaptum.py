from transformers import BartTokenizer, BartForConditionalGeneration, utils
import pdb
import torch
from captum.attr import LayerConductance, LayerIntegratedGradients

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
bart = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)

def bart_forward_func(encoder_input_ids,decoder_input_ids,index,embed):
    encoder_embeds = embed(encoder_input_ids)
    decoder_embeds = embed(decoder_input_ids)
    outputs = bart(inputs_embeds=encoder_embeds,decoder_inputs_embeds=decoder_embeds)
    pred_idx = decoder_input_ids[0,index]
    pred = outputs.logits[:,index-1,pred_idx]
    return pred

def generate_ref_sequences(input_ids):
    ref_input_ids = torch.zeros_like(input_ids)
    ref_input_ids[0,0] = tokenizer.bos_token_id
    ref_input_ids[0,1:-1] = tokenizer.pad_token_id
    ref_input_ids[0,-1] = tokenizer.eos_token_id
    return ref_input_ids

encoder_input_ids = tokenizer("The House Budget Committee voted Saturday to pass a $3.5 trillion spending bill", return_tensors="pt", add_special_tokens=True).input_ids
decoder_input_ids = tokenizer("The House Budget Committee passed a spending bill.", return_tensors="pt", add_special_tokens=True).input_ids

pdb.set_trace()
embed = torch.nn.Embedding(bart.model.shared.num_embeddings,bart.model.shared.embedding_dim,bart.model.shared.padding_idx)
embed.weight.data = bart.model.shared.weight.data
pred = bart_forward_func(encoder_input_ids,decoder_input_ids,3,embed)
ref_encoder_inputs = generate_ref_sequences(encoder_input_ids)
lig = LayerIntegratedGradients(bart_forward_func,bart.model.encoder.layers[0])
attributions_encoder = lig.attribute(inputs=(encoder_input_ids),baselines = (ref_encoder_inputs),additional_forward_args= (decoder_input_ids,2,embed),attribute_to_layer_input=False)
