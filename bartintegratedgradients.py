from transformers import BartTokenizer, BartForConditionalGeneration, utils
import pdb
import torch
from torch.utils.data import Dataset,DataLoader
import modelutils



# Bart Model Wrapper to take in encoder and decoder embeddings and output logit
# prediction scores for token of choice
# Inputs:
# bart: BartForConditionalGeneration; Bart conditional generation model 
# encoder_embeds: Torch.Tensor; Encoder Embeddings of shape (batch_size,encoder_seq_len,embed_dim)
# decoder_embeds: Torch.Tensor; Decoder Embeddings of shape (batch_size,decoder_seq_len,embed_dim)
# index: Int; Index of token for which to return prediction logits
# pred_idx: Int; Vocab index for word of interest
#
# Returns:
# pred: Torch.Tensor; Contains logits for prediction score of selected word of interest at desired position index
def bart_forward_func(bart,encoder_embeds,decoder_embeds,index,pred_idx):
    outputs = bart(inputs_embeds=encoder_embeds,decoder_inputs_embeds=decoder_embeds)
    pred = outputs.logits[:,index-1,pred_idx]
    return pred


# Bart Model Wrapper to take in encoder output and decoder embeddings and output logit
# prediction scores for token of choice
# Inputs:
# bart: BartForConditionalGeneration; Bart conditional generation model 
# encoder_output_embeds: Torch.Tensor; Encoder Output Embeddings of shape (batch_size,encoder_seq_len,embed_dim)
# decoder_embeds: Torch.Tensor; Decoder Embeddings of shape (batch_size,decoder_seq_len,embed_dim)
# index: Int; Index of token for which to return prediction logits
# pred_idx: Int; Vocab index for word of interest
#
# Returns:
# pred: Torch.Tensor; Contains logits for prediction score of selected word of interest at desired position index 
def bart_decoder_forward_func(bart,encoder_output_embeds,decoder_embeds,index,pred_idx):
    outputs = bart(encoder_outputs=(encoder_output_embeds,None,None),decoder_inputs_embeds=decoder_embeds)
    pred = outputs.logits[:,index-1,pred_idx]
    return pred    

# Bart Model Wrapper to take in encoder and decoder embeddings and output last encoder hidden state
# Inputs:
# bart: BartForConditionalGeneration; Bart conditional generation model 
# encoder_output_embeds: Torch.Tensor; Encoder Output Embeddings of shape (batch_size,encoder_seq_len,embed_dim)
# decoder_embeds: Torch.Tensor; Decoder Embeddings of shape (batch_size,decoder_seq_len,embed_dim)
#
# Returns:
# encoder_last_hidden_state: Torch.FloatTensor of shape (batch_size, sequence_length, hidden_size); Sequence of hidden-states at the output of the last layer of the encoder of the model.
def bart_enc_out_forward_func(bart,encoder_embeds,decoder_embeds):
    outputs = bart(inputs_embeds=encoder_embeds,decoder_inputs_embeds=decoder_embeds,output_hidden_states=True)
    encoder_last_hidden_state = outputs.encoder_last_hidden_state
    return encoder_last_hidden_state


# Creates dataset to return batches of path embeddings and corresponding inputs
# Necessary so that all step sizes can be accomodated on GPU
class RepEmbedsDataset(Dataset):
    def __init__(self,path_embeds_rep,other_input_embeds_rep):
        self.path_embeds = path_embeds_rep
        self.other_input_embeds = other_input_embeds_rep
    
    def __len__(self):
        return len(self.path_embeds)

    def __getitem__(self,idx):
        return self.path_embeds[idx],self.other_input_embeds[idx]



class IntegratedGradients:

    # Initializes Integrated Gradients Class
    # Inputs:
    # model: Model that will be passed into forward_func and analyzed
    # tokenizer: Relevant tokenizer for model
    # encoder_embed: nn.Embedding; Embedding layer to convert between encoder input ids and embeddings
    # decoder_embed: nn.Embedding; Embedding layer to convert between decoder input ids and embeddings    
    # forward_func: function; Compatible forward function that takes in model,encoder embeddings, decoder embeddings, token index to analyze for, and vocab_index of token index  
    # decoder_only_forward_func; Compatible forward function that takes in model,encoder output, decoder embeddings, token index to analyze for, and vocab_index of token index
    # device: Torch.device; Device for which to run calculations over
    def __init__(self,model,tokenizer,encoder_embed,decoder_embed,forward_func,decoder_only_forward_func,enc_out_forward_func,device):
        self.model = model
        self.device = device        
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.encoder_embed = encoder_embed.to(self.device)
        self.decoder_embed = decoder_embed.to(self.device)
        self.forward_func = forward_func
        self.decoder_only_forward_func = decoder_only_forward_func
        self.enc_out_forward_func = enc_out_forward_func


    # Calculates Integrated Gradient attribution scores on a Conditional Generation Model for a given decoder token
    # Can calculate attributions with respect to encoder inputs or decoder inputs
    # Integrated Gradients paper can be found here https://arxiv.org/pdf/1703.01365.pdf
    # Inputs:
    # encoder_input_ids: torch.Tensor; Tensor of shape (1,encoder_seq_len) containing encoder input ids
    # decoder_input_ids: torch.Tensor; Tensor of shape (1,decoder_seq_len) containing decoder input ids
    # index: int; Integer representing index of decoder token to analyze
    # step_size: int; Number of steps to be used for Riemann Approximation
    # attrib_mode: str; Flag for whether to produce attributions for encoder inputs,encoder outputs or decoder inputs. Valid values are "dec","enc_in",and "enc_out" 
    # debug: bool; Flag for printing out extra validation message
    #
    # Returns:
    # attributions: torch.Tensor; Tensor of shape (1,attr_seq_len) containing attribution scores for each token in either the encoder inputs or decoder inputs (depending on attribute_decoder)
    def cond_gen_integrated_gradients(self,encoder_input_ids,decoder_input_ids,index,step_size=300,attrib_mode="dec",debug=False):        
        
        #Generate Input Embeddings and Baseline Embeddings depending on
        #if we want to attribute the encoder or decoder
        with torch.no_grad():
            encoder_input_ids = encoder_input_ids.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
            pred_idx = decoder_input_ids[0,index]    
            encoder_embeds = self.encoder_embed(encoder_input_ids)
            decoder_embeds = self.decoder_embed(decoder_input_ids[:,:index])

            validate_input_args = (self.model,encoder_embeds,decoder_embeds,index,pred_idx)
            forward_func = self.forward_func

            if attrib_mode == "dec":
                bas_embeds = self.generate_baseline_embeddings(decoder_input_ids[:,:index],self.decoder_embed)
                validate_baseline_args = (self.model,encoder_embeds,bas_embeds,index,pred_idx)
                attribute_embeds = decoder_embeds
                other_input_embeds = encoder_embeds
            elif attrib_mode == "enc_in":
                bas_embeds = self.generate_baseline_embeddings(encoder_input_ids,self.encoder_embed)
                validate_baseline_args = (self.model,bas_embeds,decoder_embeds,index,pred_idx)        
                attribute_embeds = encoder_embeds
                other_input_embeds = decoder_embeds
            elif attrib_mode == "enc_out":
                encoder_last_hidden_state = self.enc_out_forward_func(self.model,encoder_embeds,decoder_embeds)
                bas_embeds = torch.zeros_like(encoder_last_hidden_state)
                validate_baseline_args = (self.model,bas_embeds,decoder_embeds,index,pred_idx)        
                attribute_embeds = encoder_last_hidden_state
                other_input_embeds = decoder_embeds    
                forward_func = self.decoder_only_forward_func            

            #generate straight line path embeddings between baseline and input
            all_path_embeds = self.generate_path_embeddings(attribute_embeds,bas_embeds,step_size)
            all_other_input_embeds_rep = other_input_embeds.repeat(step_size,1,1)
        embed_dataset = RepEmbedsDataset(all_path_embeds,all_other_input_embeds_rep)
        embed_dataloader = DataLoader(embed_dataset,batch_size=32)

        #Run path embeddings through model and accumulate gradients
        #pdb.set_trace()
        path_grads = []
        for path_embeds_batch,other_input_embeds_batch in embed_dataloader:
            path_embeds_batch.requires_grad=True      
            other_input_embeds_batch.requires_grad=True
            encoder_embeds_rep = other_input_embeds_batch if attrib_mode == "dec" else path_embeds_batch
            decoder_embeds_rep = path_embeds_batch if attrib_mode == "dec" else other_input_embeds_batch

            preds = forward_func(self.model,encoder_embeds_rep,decoder_embeds_rep,index,pred_idx)
            preds.backward(torch.ones_like(preds))
            path_grads.append(path_embeds_batch.grad.cpu())
        path_grads = torch.concat(path_grads)

        attributions = self.accumulate_grads(attribute_embeds.cpu(),bas_embeds.cpu(),path_grads,step_size)

        if debug:
            validation_diff = self.validate_attribution_sum(validate_input_args,validate_baseline_args,forward_func)
            attrib_sum = attributions.sum().item()
            print(f"If these two values aren't close (within 5%), either there is a bug or step size needs to increase: {validation_diff} == {attrib_sum}")
            print(f"Percent Difference: {abs(attrib_sum-validation_diff)/(sum([attrib_sum,validation_diff])/2)}")
        

        #summarize attributions
        attributions = attributions.sum(dim=-1)
        attributions = attributions/torch.norm(attributions)

        return attributions


    # Calculates value of sum of path integral for verification purposes by subtracting forward function output on baseline from input
    # Based on Proposition 1 from https://arxiv.org/pdf/1703.01365.pdf
    # Inputs: 
    # input_forward_func_args: Tuple; Tuple containing function arguments for model's forward function that will produce output on input
    # baseline_forward_func_args: Tuple; Tuple containing function arguments for model's forward function that will produce output on baseline
    #
    # Returns:
    # validation_diff: int; Integer representing the difference described in Proposition 1 discussed above
    def validate_attribution_sum(self,input_forward_func_args,baseline_forward_func_args,forward_func):
        input_pred = forward_func(*input_forward_func_args)
        baseline_pred = forward_func(*baseline_forward_func_args)
        validation_diff = input_pred-baseline_pred
        return validation_diff.item()

    # Calculates path integral using equation 3 from section five of https://arxiv.org/pdf/1703.01365.pdf
    # Inputs:
    # x: Torch.Tensor; Input Embeddings attributions are being generated for of shape (1, seq_len, embed_dim);
    # x_prime: Torch.Tensor; Baseline Embeddings corresponding to Input Embeddings of shape (1, seq_len, embed_dim);
    # grads: Torch.Tensor; Gradients of straight line path embeddings from x_prime to x with respects to x. Has shape (m,seq_len,embed_dim)
    # m: int; Integer signifying number of steps to be taken during Riemann Approximation of path integral
    def accumulate_grads(self,x,x_prime,grads,m=300):
        x_minus_x_prime = x-x_prime
        grads = grads*(1/m)
        grads = torch.sum(grads,0,keepdim=True)
        attributions = x_minus_x_prime*grads
        return attributions

    # Generates the straight line path between the input embeddings x and the baseline embeddings x_prime
    # using the numerator of the derivative from equation 3 from section five of https://arxiv.org/pdf/1703.01365.pdf
    # Inputs:
    # x: Input Embeddings attributions are being generated for of shape (1, seq_len, embed_dim); Torch.Tensor
    # x_prime: Baseline Embeddings corresponding to Input Embeddings of shape (1, seq_len, embed_dim); Torch.Tensor
    # m: Integer signifying number of steps to be taken during Riemann Approximation of path integral
    #
    # Returns
    # path_embeddings: Straight Line Path Embeddings between Baseline Embeddings and Input Embeddings. Shape (m, seq_len, embed_dim)
    def generate_path_embeddings(self,x,x_prime,m=300):

        x_minus_x_prime = x-x_prime
        k_over_m = (torch.arange(m,dtype=x.dtype,device=x.device)+1)/m

        x_minus_x_prime = x_minus_x_prime.repeat(m,1,1)
        x_prime = x_prime.repeat(m,1,1)
        k_over_m = k_over_m[:,None,None]

        path_embeddings = x_prime+k_over_m*x_minus_x_prime
        return path_embeddings


    # Generates the reference embeddings for a given set of input ids using BART special token scheme
    # Inputs: 
    # input_ids: Torch.Tensor of shape(1,seq_len) 
    # embed: nn.Embedding layer to convert ids into embeddings of shape embed_dim
    #
    # Returns
    # bas_embeddings: Torch.Tensor; Baseline Embeddings of shape (1,seq_len, embed_im)
    def generate_baseline_embeddings(self,input_ids,embed):
        bas_input_ids = torch.zeros_like(input_ids)
        bas_input_ids[0,0] = self.tokenizer.bos_token_id
        bas_input_ids[0,1:-1] = self.tokenizer.pad_token_id
        bas_input_ids[0,-1] = self.tokenizer.eos_token_id
        bas_embeddings = embed(bas_input_ids)
        return bas_embeddings


def main():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    bart = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)

    encoder_input_ids = tokenizer("The House Budget Committee voted Saturday to pass a $3.5 trillion spending bill", return_tensors="pt", add_special_tokens=True).input_ids
    decoder_input_ids = tokenizer("The House Budget Committee passed a spending bill.", return_tensors="pt", add_special_tokens=True).input_ids

    #pdb.set_trace()
    embed = torch.nn.Embedding(bart.model.shared.num_embeddings,bart.model.shared.embedding_dim,bart.model.shared.padding_idx)
    embed.weight.data = bart.model.shared.weight.data
    analyze_idx = 5

    device = torch.device("cuda")
    integrated_gradients = IntegratedGradients(bart,tokenizer,embed,embed,bart_forward_func,bart_decoder_forward_func,bart_enc_out_forward_func,device)
    print("Calculating Decoder Attributions")
    decoder_attributions = integrated_gradients.cond_gen_integrated_gradients(encoder_input_ids,decoder_input_ids,analyze_idx,attrib_mode="dec",step_size=1000,debug=True)
    print("Calculating Encoder Input Attributions")
    encoder_input_attributions = integrated_gradients.cond_gen_integrated_gradients(encoder_input_ids,decoder_input_ids,analyze_idx,attrib_mode="enc_in",debug=True)
    print("Calculating Encoder Output Attributions")
    encoder_output_attributions = integrated_gradients.cond_gen_integrated_gradients(encoder_input_ids,decoder_input_ids,analyze_idx,attrib_mode="enc_out",debug=True)


    decoder_text = modelutils.replace_special_bart_tokens(modelutils.get_id_text(decoder_input_ids,tokenizer)) 
    decoder_text = decoder_text[:analyze_idx]

    encoder_text = modelutils.replace_special_bart_tokens(modelutils.get_id_text(encoder_input_ids,tokenizer)) 


    print("\nDecoder Attributions")
    concat_attr_text_score =  [ i+": ,"+str(j) for i,j in zip(decoder_text,decoder_attributions[0].tolist())]
    print("".join(concat_attr_text_score))

    print("\nEncoder Input Attributions")
    concat_attr_text_score =  [ f"{i}: {j:.4f}\n" for i,j in zip(encoder_text,encoder_input_attributions[0].tolist())]
    print("".join(concat_attr_text_score))

    print("\nEncoder Output Attributions")
    concat_attr_text_score =  [ f"{i}: {j:.4f}\n" for i,j in zip(encoder_text,encoder_output_attributions[0].tolist())]
    print("".join(concat_attr_text_score))    


if __name__ == "__main__":
    main()



