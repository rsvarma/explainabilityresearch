from transformers import BartTokenizer, BartForConditionalGeneration, utils
import pdb
import json
import modelutils
import torch
from captum.attr import LayerConductance, LayerIntegratedGradients

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)



class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


f = open('annotations.jsonl','r')
lines = f.readlines()
of = open('intrinsicattentionanalysis.txt','w',encoding='UTF-8')

for line in lines:
    data = json.loads(line)
    if "intrinsic" in data['bart']['label']:
        if data['media_source'] == 'cnndm':

            intrinsic_idx_list = modelutils.get_intrinsic_indices(data['bart']['label'])
            #pdb.set_trace()
            decoder_words = data['bart']['words']
            decoder_words = modelutils.join_words(decoder_words)             
            encoder_input_ids,decoder_input_ids = modelutils.get_input_ids(data['document'],decoder_words,tokenizer)
            encoder_text = modelutils.get_id_text(encoder_input_ids,tokenizer)
            decoder_text = modelutils.get_id_text(decoder_input_ids,tokenizer)                        
            intrinsic_id_indices = modelutils.get_error_id_indices(data['bart']['label'],data['bart']['words'],tokenizer,"intrinsic")
            intrinsic_idx_list = [ i for i in range(len(data['bart']['label'])) if data['bart']['label'][i] == 'intrinsic']
            decoder_words = data['bart']['words']
            for i in intrinsic_idx_list:
                decoder_words[i] = color.RED+data['bart']['words'][i]+color.END
            decoder_words = modelutils.join_words(decoder_words)                             
            print(data['document'],file=of)
            print("\n",file=of)
            print(decoder_words,file=of)
            print("\n",file=of)
            print(decoder_text,file=of)
            print("\n",file=of)
            #pdb.set_trace()
            lig1 = LayerIntegratedGradients(modelutils.bart_forward_func,model.model.decoder.embed_tokens)
            lig2 = LayerIntegratedGradients(modelutils.bart_forward_func,model.model.encoder.embed_tokens)
            lig3 = LayerIntegratedGradients(modelutils.bart_forward_func,model.model.encoder.layernorm_embedding)            
            
            ref_encoder_inputs = modelutils.generate_ref_sequences(encoder_input_ids,tokenizer)            
            ref_decoder_inputs = modelutils.generate_ref_sequences(decoder_input_ids,tokenizer)
            
            for i in intrinsic_id_indices:
                print(f"Analyzing for intrinsic error token:{decoder_text[i]}\n",file=of)
                outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
                cross_attentions_list = outputs.cross_attentions
                print("Top 5 Cross Attention",file=of)                
                for layer_id,cross_attentions in enumerate(cross_attentions_list):
                    layer_score = torch.mean(cross_attentions[:,:,i,:].squeeze(0),dim=0)
                    top_indices = torch.topk(layer_score,5).indices.tolist()
                    top_text = [encoder_text[idx] for idx in top_indices]
                    print(top_text,file=of)    
                print("\n")
                print("Top 5 Decoder Attention",file=of)                
                decoder_attentions_list = outputs.decoder_attentions
                for layer_id,decoder_attentions in enumerate(decoder_attentions_list):
                    layer_score = torch.mean(decoder_attentions[:,:,i,:].squeeze(0),dim=0)
                    top_indices = torch.topk(layer_score,5).indices.tolist()
                    top_text = [decoder_text[idx] for idx in top_indices]
                    print(top_text,file=of)    
                
                print("\n")
                print(f"Analyzing for token before intrinsic error:{decoder_text[i-1]}\n",file=of)
                outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
                cross_attentions_list = outputs.cross_attentions
                print("Top 5 Cross Attention",file=of)                   
                for layer_id,cross_attentions in enumerate(cross_attentions_list):
                    layer_score = torch.mean(cross_attentions[:,:,i-1,:].squeeze(0),dim=0)
                    top_indices = torch.topk(layer_score,5).indices.tolist()
                    top_text = [encoder_text[idx] for idx in top_indices]
                    print(top_text,file=of)    
                print("\n\n",file=of)
                decoder_attentions_list = outputs.decoder_attentions
                print("Top 5 Decoder Attention",file=of)                 
                for layer_id,decoder_attentions in enumerate(decoder_attentions_list):
                    layer_score = torch.mean(decoder_attentions[:,:,i-1,:].squeeze(0),dim=0)
                    top_indices = torch.topk(layer_score,5).indices.tolist()
                    top_text = [decoder_text[idx] for idx in top_indices]
                    print(top_text,file=of)                 

                print("\n\n",file=of)
                #attributions_encoder = lig2.attribute(inputs=(encoder_input_ids),baselines = (ref_encoder_inputs),additional_forward_args= (decoder_input_ids,model,i))

                
                 

                             
           

