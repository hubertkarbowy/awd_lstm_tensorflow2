import argparse
import logging
import sentencepiece as spm

"""
Trains SPM model

python ./02_build_spm.py \
        --corpus-path $LM_MODELS/../datasets/books/therepublic_pretokenized.txt \
        --vocab-size 5000 \
        --model-prefix plato

"""

logging.basicConfig(level=logging.INFO)

def main(args):
    logging.info(f"Reading input file {args['corpus_path']}")
    model_prefix = f"{args['model_prefix']}-sp{args['vocab_size']//1000}k"
    logging.info(f"Fitting the SPM tokenizer - will save to {model_prefix}")
    spm.SentencePieceTrainer.train(input=args['corpus_path'], vocab_size=args['vocab_size'], \
                                   unk_id=0, pad_id=1, bos_id=2, eos_id=3, model_prefix=model_prefix, \
                                   model_type='bpe', character_coverage=0.995, num_threads=4)
    print("Training completed!. Now run the 02b_encode_spm.py script to convert your corpus to token ids.")
    print("Alternatively, you can use spm_encode as follows (you won't get corpus stats, though):")
    print()
    print(f"1. To obtain word pieces (for sanity check, usually not needed for training):")
    print(f"   $ spm_encode --model ./{model_prefix}.model \\\n" \
           "     input_corpus.txt --extra_options=\"bos:eos\" > tokenized_train_pieces.txt")
    print(f"2. To obtain piece ids: (needed for training)")
    print(f"   $ spm_encode --model ./{model_prefix}.model \\\n" \
           "     input_corpus.txt --extra_options=\"bos:eos\" --output-format id > tokenized_train_ids.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True, help="Path to a raw text corpus (cleaned up and preprocessed). One big single file.")
    parser.add_argument("--vocab-size", required=True, type=int, help="How big should be the SPM vocab")
    parser.add_argument("--model-prefix", required=False, default="customspm", help="SPM model prefix")
    argz = parser.parse_args()
    main(vars(argz))