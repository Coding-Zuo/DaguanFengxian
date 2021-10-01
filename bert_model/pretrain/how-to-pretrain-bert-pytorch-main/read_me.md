## how to pretrain bert

## 1.prepare your data
# run process_data.py

single sentence:
i like go to school, and i like study .

sentence pair:
i like to to school, \t  and i like study .
(use tab to sep sentence pair) 

## 2.build vocab
# run build_vocab.py

use processed data to build .

## 3.pretrain bert
# run run_pretrain.py