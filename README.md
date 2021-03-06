# Sentiment Analysis on Yelp Reviews.
10/15/2018 update: epoch_size=10, batch_size=128.
Test accuracy: 88%

## Data:
Data set is too large to upload on Github. Data could be found [here](https://www.yelp.com/dataset).
I use *yelp_academic\_dataset\_review.json* in this repo.

A new bash script is added to split data. Run the scripts to split data into pieces by years

```{bash}
$ cat scripts/years.txt | xargs -I{} bash -c "bash scripts/split_json.sh {}"
```

## Convert _\.json_ to _\.tsv_
Converting *yelp_academic\_dataset\_review.json* to *yelp_academic\_dataset\_review.tsv*.

```{bash}
ls ./data/*201*json | xargs -n 1 -I{} bash -c "python json2tsv.py --json={}"
```

## Training:
```{bash}
$ python train.py [-h] [-b BATCH_SIZE] [-e EPOCH_SIZE] [-l SEQ_LENGTH]
                [-c CHECK_DIR]
```

* List of optional arguments

```
optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Set the batch size during training
  -e EPOCH_SIZE, --epoch_size EPOCH_SIZE
                        Set the epoch size during training
  -l SEQ_LENGTH, --seq_length SEQ_LENGTH
                        The length to which each sequence will be converted
  -c CHECK_DIR, --check_dir CHECK_DIR
                        The path where the model will be restored
```

* Example

Train the model while setting 8 epochs and 128 records per batch.

```{bash}
$ python train.py --epoch_size=8 --batch_size=128
```

## Visualization

```
tensorboard --logdir='path/to/your/summary'
```
