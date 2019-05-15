# This file is to run experiments with Context Agnostic Generative (CAG) and the Context Informed (CIG) model
# Please use the code base in https://github.com/shrimai/Style-Transfer-Through-Back-Translation git repo to run the commands given in this file

# preporcess data for the CAG model
python preprocess.py -train_src train_data/train.src -train_tgt train_data/train.tgt -valid_src train_data/valid.src -valid_tgt train_data/valid.tgt -src_vocab_size  32000 -tgt_vocab_size 32000 -seq_length 2000 -lower $true -save_data data/cag_data 

# train CAG model
python nmt_train.py -data data/cag_data.train.pt -save_model cag_model/cag_model -layers 2 -epochs 30 -brnn $true -rnn_size 128 -word_vec_size 100 -gpus 0 -start_decay_at 10 

# generate using the CAG model
python translate.py -gpu 0 -encoder_model cag_model/cag_model.pt -decoder_model cag_model/cag_model.pt -replace_unk $true -src train_data/test.src -output cag_model/cag_model_test.out

# preprocess data for the CIG model
python preprocess.py -train_src train_data/train.srcxt -train_tgt train_data/train.tgt -valid_src train_data/valid.srcxt -valid_tgt train_data/valid.tgt -src_vocab_size  32000 -tgt_vocab_size 32000 -seq_length 2500 -lower $true -save_data data/cig_data 

# train CIG model
python nmt_train.py -data data/cig_data.train.pt -save_model cig_model/cig_model -layers 2 -epochs 30 -brnn $true -rnn_size 128 -word_vec_size 100 -gpus 0 -start_decay_at 10 

# generate using the CIG model
python translate.py -gpu 0 -encoder_model cig_model/cig_model.pt -decoder_model cig_model/cig_model.pt -replace_unk $true -src train_data/test.srcxt -output cig_model/cig_model_test.out

# After generating output, you might want to decode the ouput using the trained sentencepiece model. The intstructions are given on the main page.
