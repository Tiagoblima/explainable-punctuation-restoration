DATASET_NAME=tiagoblima/punctuation-tedtalk2012-bert
EMBEDDING_PATH=/content/punctuation-verification/punctuation-restoration/blistm/embeddings/skip_s300.gensim

cd data/

python build_dataset.py \
      --dataset_name $DATASET_NAME

cd ../


python run_flair.py \
    --path_to_data ./data/datasets/tiagoblima/punctuation-tedtalk2012-bert \
    --embeddings_bin_file $EMBEDDING_PATH \
    --model_dir '/content/drive/MyDrive/Informática Aplicada/Pesquisa/models/'