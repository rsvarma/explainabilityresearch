
from os.path import join,normpath
import itertools
import pdb

def add_beginning_template(title):
    html_string = f'''<!doctype html>

    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>{title}</title>
    <link rel="stylesheet" href="../../css/styles.css?v=1.0">

    </head>

    <body>
    '''

    return html_string

def add_end_template():
    return f'''
    </body>
    </html>    
    '''

def print_tokens(token_list,error_id_indices=None,print_error_color=False):
    html_string = '  <div class="token-container">\n'
    for i,token in enumerate(token_list):
        if token == "<s>":
            token = "begin_sequence"
        elif token == "</s>":
            token = "end_sequence"

        if print_error_color:
            if i in error_id_indices:
                html_string += f'    <div class="token-tag" style="color:red;"><a href="{token}{i}.html">{token}</a> &#10; {i}</div>\n'
            else:
                html_string += f'    <div class="token-tag"><a href="{token}{i}.html">{token}</a> &#10; {i}</div>\n'                

        else:          
            html_string += f'    <div class="token-tag">{token} &#10; {i}</div>\n'
    html_string += f'  </div>\n'
    return html_string


def print_attentions(cross_vals,cross_inds,cross_text,dec_vals,dec_inds,dec_text):
    html_string = '  <div class="outer-attention-container">\n'
    html_string += '  <div class="attention-container">\n'
    html_string += '  <h2>Cross Attentions</h2>\n'    
    for value,index,token in zip(cross_vals,cross_inds,cross_text) :
        if token == "<s>":
            token = "begin_sequence"
        elif token == "</s>":
            token = "end_sequence"         
        html_string += f'    <div class="token-tag">{token} &#10;Attention Score:{value} &#10;Index:{index}</div>\n'
    html_string += f'  </div></div>\n'    
    html_string += '  <div class="outer-attention-container">\n'
    html_string += '  <div class="attention-container">\n'
    html_string += '  <h2>Decoder Attentions</h2>\n'
    for value,index,token in zip(dec_vals,dec_inds,dec_text) :
        if token == "<s>":
            token = "begin_sequence"
        elif token == "</s>":
            token = "end_sequence"         
        html_string += f'    <div class="token-tag">{token} &#10;Attention Score:{value} &#10;Index:{index}</div>\n'
    html_string += f'  </div></div>\n'     
    return html_string


def print_example_page(dir_path,title,encoder_tokens,decoder_tokens,error_id_indices):
    of = open(normpath(join(dir_path,f'{title}.html')),'w',encoding='UTF-8')
    html_string = add_beginning_template(title)
    html_string += "<h2>Encoder Tokens </h2>\n"
    html_string += print_tokens(encoder_tokens)
    html_string += "<h2> Decoder Tokens</h2>\n"
    html_string += print_tokens(decoder_tokens,error_id_indices,True)
    html_string += add_end_template()
    print(html_string,file=of)
    of.close()

    
def print_token_page(dir_path,title,attentions):
    of = open(normpath(join(dir_path,f'{title}.html')),'w',encoding='UTF-8')
    html_string = add_beginning_template(title)
    html_string += f"<h1>Analysis for {title}</h1>\n"
    for i in range(len(attentions)):
        html_string += '<div class="wrap-collapsible">\n'
        html_string += f'<input id="collapsible{i}" class="toggle" type="checkbox">\n'
        html_string += f'<label for="collapsible{i}" class="lbl-toggle">Layer {i+1}</label>\n'
        html_string += f'<div class="collapsible-content">\n'
        print_args = [[v for (k,v) in att_dict.items()] for (key,att_dict) in attentions[i].items()]
        print_args = list(itertools.chain.from_iterable(print_args))
        html_string += print_attentions(*print_args)
        html_string += '</div></div>\n'
    print(html_string,file=of)
    of.close()

 

        

