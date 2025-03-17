import string
from pathlib import Path

import tokenizers.normalizers
from tokenizers import Tokenizer


def load_texts(name_dir: Path) -> list[str]:
    all_name_files = [name_dir / 'English.txt', name_dir / 'German.txt', name_dir / 'Italian.txt']
    all_names = []
    for name_file in all_name_files:
        all_names.extend(name_file.read_text(encoding='utf-8').splitlines())
    return all_names


def train():
    texts = load_texts(Path('/home/me/projects/misis-2526/data/names'))
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
        special_tokens=[('[SOS]', 0), ('[EOS]', 1)]
    )

    trainer = tokenizers.trainers.WordPieceTrainer(
        vocab_size=512,
        show_progress=True,
        special_tokens=['[SOS]', '[EOS]', '[UNK]', '[PAD]'],
        initial_alphabet=list(string.ascii_letters + string.digits),
        continuing_subword_prefix='##'
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.save('/home/me/projects/misis-2526/saved_tokenizers/tokenizer.json', pretty=True)


if __name__ == '__main__':
    train()
