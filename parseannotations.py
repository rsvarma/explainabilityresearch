import json


def make_long_sentence_readable(sentence):
    words = sentence.split(' ')
    newline_idxs = [i for i in range(29,len(words),30)]
    for i in newline_idxs:
        words[i] = words[i]+'\n'
    return words

def join_words(words):
    str = ""
    for word in words:
        if word == "-":
            str = str[:-1]+word
        if word == "." or word == "," or word == "?":
            str = str[:-1]+word+" "
        else:
            str += word+" "
    return str



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

out_file = open('intrinsicerrs.txt','w')
f= open('annotations.jsonl','r')
lines = f.readlines()
for line in lines:
    data = json.loads(line)
    if "intrinsic" in data['bart']['label']:
        if data['media_source'] == 'cnndm':
            intrinsic_idx_list = [ i for i in range(len(data['bart']['label'])) if data['bart']['label'][i] == 'intrinsic']
            for i in intrinsic_idx_list:
                data['bart']['words'][i] = color.RED+data['bart']['words'][i]+color.END
            summary =  join_words(data['bart']['words'])
            document = join_words(make_long_sentence_readable(data['document']))
            #document = data['document']
            out_file.write('DOCUMENT:\n')
            out_file.write(document+'\n\n\n')
            out_file.write('SUMMARY\n')
            out_file.write(f"{summary}\n\n\n")


            


    
            