from tokenizers import Tokenizer

if __name__ == '__main__':
    tokenizer = Tokenizer.from_file('/home/me/projects/misis-2526/saved_tokenizers/tokenizer.json')
    result = tokenizer.encode('Maxim')
    print(result.tokens)
    print(result.ids)