# Towards Content Transfer through Grounded Text Generation

This repo contains the code and data of the following paper:
>Towards Content Transfer through Grounded Text Generation. *Shrimai Prabhumoye, Chris Quirk, Michel Galley*. NAACL 2019. [arXiv](http://arxiv-export-lb.library.cornell.edu/pdf/1905.05293)

## Dependencies

- Python 3.6
- Pytorch 0.3
- [sentencepiece](https://github.com/google/sentencepiece/blob/master/python/README.md)
- [NLTK](https://www.nltk.org/install.html)
- nltk.download("stopwords") in your python terminal.

## Data

Dowload the train, dev, and test data for all the experiments from the following link: 
```bash
http://tts.speech.cs.cmu.edu/content_transfer/train_data.zip
unzip train_data.zip
```
The \*.src files contain the news articles, \*.cxt files contain the Wikipedia context, \*.tgt files contain the target sentences and the \*.srcxt files contain the news articles concatenated with Wikipedia context used in CAG models.

Dowload the raw data for train, dev, and test splits from the following link:
```bash
http://tts.speech.cs.cmu.edu/content_transfer/raw_data.zip
unzip raw_data.zip
```
The raw data gives the following information: 
1. wikiID: Wikipedia page ID
2. wikiTitle: Wikipedia page Title
3. wikiContext: Context of the Wikipedia article as is. This is a list of list of sentences. 
4. Target: Target sentence from the Wikipedia article as is.
5. clean_wikiContext: The cleaned version of the Wikipedia context. This is a list of sentences.
6. clean_Target: The cleaned version of the target sentence.
7. domain: The domain of the news article.
8. URL: The URL of the news article.
9. curlCommand: The curl command to download the news article from common crawl.
10. HTML_Text: HTML of the news article converted to plain text.
11. clean_HTML_Text: Clean version of the plain text.

The ```domains.txt``` file contains the list of domains used to collect the dataset.

## Models

- ### Trained sentencepiece model

Download the trained sentencepeice model used in all experiments.
```bash
http://tts.speech.cs.cmu.edu/content_transfer/sentencepieceModel.zip
unzip sentencepieceModel.zip
```

To train sentencepiece model on your data:
```bash
python sentence_piece.py -mode train -input sentencepieceModel/train.data -model_prefix testModel -model_type bpe -vocab_size 32000
```

To encode data using the trained sentencepiece model:
```bash
python sentence_piece.py -mode encode -input inputFilename.txt -model sentencepieceModel/bpeM.model -output outputFilename.txt
```

To decode the generated data using the trained sentencepiece model:
```bash
python sentence_piece.py -mode decode -input inputFilename.txt -output outputFilename.txt -model sentencepieceModel/bpeM.model
```

- ### Sum-Basic(SB) and Context Informed Sum-Basic (CISB)
```bash
python sumbasicUpdate.py -input raw_data/filename.csv -output filename.txt
```

Use the ```-context_update``` flag for CISB.

- ### Context Agnostic Generative (CAG) Model and Context Informed Generative (CIG) Model

Please use the code base in the following git repo for these two models:
https://github.com/shrimai/Style-Transfer-Through-Back-Translation
Refer to example.sh file to see the commands.

Download the trained CAG model:
```bash
http://tts.speech.cs.cmu.edu/content_transfer/cag_model.zip
unzip cag_model.zip
```

Download the trained CIG model:
```bash
http://tts.speech.cs.cmu.edu/content_transfer/cig_model.zip
unzip cig_model.zip
```

- ### Context Receptive Generative (CRG) Model

Follow the example.sh file in the ```context_receptive_generative/``` directory.

Download the trained CRG model:
```bash
http://tts.speech.cs.cmu.edu/content_transfer/crg_model.zip
unzip crg_model.zip
```

If you are using this data or code then please cite the following paper::

    @inproceedings{content_transfer_naacl19,
    title={Towards Content Transfer through Grounded Text Generation},
    author={Prabhumoye, Shrimai and Quirk, Chris and Galley, Michel},
    year={2019},
    booktitle={Proc. NAACL}
    }
