#The instructions in this file are for experiments with the Context Receptive (CRG) model

# Preprocess data for the CRG model
python preprocess.py -train_src train_data/train.src -train_cxt train_data/train.cxt -train_tgt train_data/train.tgt -valid_src train_data/valid.src -valid_cxt train_data/valid.cxt -valid_tgt train_data/valid.tgt -save_data data/crg_data -src_vocab_size 32000 -tgt_vocab_size 32000 -seq_length_src 2000 -seq_length_cxt 500 -seq_length_tgt 200

# Train the CRG model
python nmt_train.py -data data/crg_data.train.pt -save_model crg_model/crg_model -word_vec_size 100 -brnn -epochs 30 -batch_size 32 -layers 2 -rnn_size 128 -gpus 0

# Generate using the CRG model
python translate.py -model crg_model/crg_model.pt -src train_data/test.src -cxt train_data/test.cxt -output crg_model/crg_model_test.out -replace_unk -gpu 0
