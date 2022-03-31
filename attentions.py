from transformers import BartTokenizer, BartForConditionalGeneration, utils
import pdb
import json
import modelutils
import torch
from captum.attr import LayerConductance, LayerIntegratedGradients
import sys
from htmlgenerator import print_example_page,print_token_page
import os

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)

def bart_forward_func(encoder_input_ids,decoder_input_ids,index):
    pdb.set_trace()
    outputs = model(input_ids=encoder_input_ids,decoder_input_ids=decoder_input_ids)
    pred_idx = decoder_input_ids[0,index]
    pred = outputs.logits[0,index-1,pred_idx]
    return pred

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
#of = open('intrinsicattentionanalysis.txt','w',encoding='UTF-8')
of = sys.stdout
for example_id,line in enumerate(lines):
    data = json.loads(line)
    if "intrinsic" in data['bart']['label']:
        if data['media_source'] == 'cnndm':

            intrinsic_idx_list = modelutils.get_intrinsic_indices(data['bart']['label'])
            decoder_words = data['bart']['words']
            decoder_words = modelutils.join_words(decoder_words)             
            encoder_input_ids,decoder_input_ids = modelutils.get_input_ids(data['document'],decoder_words,tokenizer)
            encoder_text = modelutils.replace_special_bart_tokens(modelutils.get_id_text(encoder_input_ids,tokenizer))
            decoder_text = modelutils.replace_special_bart_tokens(modelutils.get_id_text(decoder_input_ids,tokenizer))                        
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
            example_dir_path = os.path.normpath(f"./examples/intrinsicexamples/example{example_id}")
            if not os.path.exists(example_dir_path):
                os.mkdir(example_dir_path)
            print_example_page(example_dir_path,"exampletokens",encoder_text,decoder_text,intrinsic_id_indices)
            



            #pdb.set_trace()
            '''
            lig1 = LayerIntegratedGradients(modelutils.bart_forward_func,model.model.decoder.embed_tokens)
            lig2 = LayerIntegratedGradients(modelutils.bart_forward_func,model.model.encoder.embed_tokens)
            lig3 = LayerIntegratedGradients(modelutils.bart_forward_func,model.model.encoder.layernorm_embedding)            
            
            ref_encoder_inputs = modelutils.generate_ref_sequences(encoder_input_ids,tokenizer)            
            ref_decoder_inputs = modelutils.generate_ref_sequences(decoder_input_ids,tokenizer)
            '''
            
            for i in range(len(decoder_input_ids[0])):
                outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
                cross_attentions_list = outputs.cross_attentions    
                decoder_attentions_list = outputs.decoder_attentions
                attention_info_list = []                          
                for layer_id,(cross_attentions,decoder_attentions) in enumerate(zip(cross_attentions_list,decoder_attentions_list)):
                    layer_score = torch.mean(cross_attentions[:,:,i,:].squeeze(0),dim=0)
                    top_info = torch.topk(layer_score,10)
                    top_cross_values = top_info.values.tolist()
                    top_cross_indices = top_info.indices.tolist()
                    top_cross_text = [encoder_text[idx] for idx in top_cross_indices]
            

                    layer_score = torch.mean(decoder_attentions[:,:,i,:].squeeze(0),dim=0)
                    top_info = torch.topk(layer_score,10)
                    top_decoder_values = top_info.values.tolist()
                    top_decoder_indices = top_info.indices.tolist()
                    top_decoder_text = [decoder_text[idx] for idx in top_decoder_indices]


                    attention_info = {
                        "cross_attention":{"values":top_cross_values,
                                            "indices": top_cross_indices,
                                            "text": top_cross_text
                                        },
                        "decoder_attention":{"values": top_decoder_values,
                                            "indices": top_decoder_indices,
                                            "text": top_decoder_text}
                    }       
                    attention_info_list.append(attention_info)
                print_token_page(example_dir_path,f"{decoder_text[i]}{i}",attention_info_list)
                



          




                
                 

                             
           

