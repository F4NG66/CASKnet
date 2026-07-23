# CASKnet

CASKnet is a deep learning framework for predicting **small secreted proteins (SSPs, 10–110 aa)**. It combines a pre-trained ProtT5 protein language model encoder, multi-scale convolutional feature extraction, multi-head self-attention, and a Kolmogorov–Arnold Network (KAN) prediction head. Unlike classical signal-peptide predictors (SignalP, TargetP), CASKnet is trained end-to-end as a sequence-level binary classifier and generalizes well to short proteins that lack canonical N-terminal signal peptides.


You will also need:

- **A trained CASKnet checkpoint** (default expected at `./model/best_model.pth`)
- **A local copy of the ProtT5-XL-UniRef50 encoder** (default expected at `./prot5uni50full`), downloadable from the [Rostlab ProtTrans repository](https://github.com/agemagician/ProtTrans)

```

---

## 2. Predicting secretion from a FASTA file

### Basic usage

```bash
python predict.py -i input.fasta -o predictions.csv
```

### All arguments

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Path to the input FASTA file. |
| `-o`, `--output` | *(required)* | Path to save the output CSV file. |
| `--model` | `./model/best_model.pth` | Path to the trained CASKnet checkpoint. |
| `--t5_model` | `./prot5uni50full` | Path to the local ProtT5 encoder directory. |
| `--batch_size` | `4` | Number of sequences processed per batch. Increase if you have enough GPU memory. |

### Example

```bash
python predict.py \
    -i my_candidate_smorfs.fasta \
    -o my_candidate_smorfs_predictions.csv \
    --model ./model/best_model.pth \
    --t5_model ./prot5uni50full \
    --batch_size 8
```

### Input format

Standard FASTA:

```
>seq1
MSLTSSSSVRVEWIAAVTIAAGTAAIGYLAYKRFYVKDHRNKAMINLHIQKDNPKIVHAFDMEDLGDKAV
>seq2
MIWTAVIKGSALMTFVQGAMALVDKVFGEEILPHRIYSSGEAAQLLGMERLQVLEMVRAGTIKAKKVGDNYRIL
```

- Sequences are automatically cleaned (`U`, `Z`, `O` → `X`) before being passed to ProtT5, since these non-standard residues are not part of its vocabulary.
- CASKnet was trained and validated on sequences **10–110 aa** long. Sequences longer than 110 aa will still be processed, but the script prints a warning, since prediction accuracy degrades outside the training length range.

### Output format

The output CSV has four columns:

| Column | Description |
|---|---|
| `ID` | Sequence identifier, taken from the FASTA header. |
| `Sequence` | The original (uncleaned) amino acid sequence. |
| `Prediction` | Binary class label: `1` = predicted secreted, `0` = predicted non-secreted. |
| `confidence` | Predicted probability of the secreted class (softmax output), rounded to 5 decimal places. |

Example:

```csv
ID,Sequence,Prediction,confidence
seq1,MSLTSSSSVRVEWIAAVTIAAGTAAIGYLAYKRFYVKDHRNKAMINLHIQKDNPKIVHAFDMEDLGDKAV,0,0.00174
seq2,MIWTAVIKGSALMTFVQGAMALVDKVFGEEILPHRIYSSGEAAQLLGMERLQVLEMVRAGTIKAKKVGDNYRIL,1,0.99896
```

Results are written incrementally, batch by batch, so partial output is preserved even if the run is interrupted midway through a large FASTA file.

---

