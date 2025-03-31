import json
import string
from pathlib import Path

import pandas as pd
import tokenizers.normalizers
from tokenizers import Tokenizer


def load_texts(wmt_path: Path) -> list[str]:
    df = pd.read_parquet(wmt_path)
    df = df['translation']
    en_texts = df.apply(lambda x: x['en'])
    ru_texts = df.apply(lambda x: x['ru'])
    return en_texts.tolist() + ru_texts.tolist()



def train():
    texts = load_texts(Path('/home/me/downloads/train-00000-of-00001.parquet'))
    tokenizer = Tokenizer(tokenizers.models.WordPiece(
        unk_token='[UNK]'
    ))
    tokenizer.normalizer = tokenizers.normalizers.Strip()
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.decoder = tokenizers.decoders.WordPiece(
        prefix='##'
    )
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[SOS] $0 [EOS]",
        special_tokens=[('[SOS]', 1), ('[EOS]', 2)]
    )

    trainer = tokenizers.trainers.WordPieceTrainer(
        vocab_size=32000,
        show_progress=True,
        special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]'],
        initial_alphabet=list(string.ascii_letters + string.digits),
        continuing_subword_prefix='##'
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.save('/home/me/projects/misis-2526/saved_tokenizers/transformer_tokenizer.json', pretty=True)


if __name__ == '__main__':
    train()
