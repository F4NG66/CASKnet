from model.kan import KAN
# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from DataSet.xlsxPLMDataSet import xlsxProteinDataset,MySelf_collate_fullEmbed
from torch.utils.data  import Dataset, Subset ,random_split
from sklearn.model_selection  import KFold
import torch.backends
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from torch.utils.data import WeightedRandomSampler
from transformers import T5EncoderModel, T5Tokenizer
import argparse
import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from model.kan import KAN

def get_T5_model(model_dir, device):
    model = T5EncoderModel.from_pretrained(model_dir, torch_dtype=torch.float32).to(device)
    if device == torch.device("cpu"):
        model.to(torch.float32)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, tokenizer

def load_fasta(fasta_path):
    sequences = {}
    with open(fasta_path, 'r') as f:
        current_id = None
        for line in f:
            if line.startswith('>'):
                current_id = line.strip().replace(">", "")
                sequences[current_id] = ""
            else:
                sequences[current_id] += line.strip().upper().replace("-", "")
    return sequences

def batch_generator(sequences, batch_size):
    items = list(sequences.items())
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

def predict_from_fasta(fasta_path, model_path, t5_dir, output_path, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = KAN([4 * 64, 64, 2])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Load T5 encoder
    T5model, tokenizer = get_T5_model(t5_dir, device)

    # Load sequences
    sequences = load_fasta(fasta_path)
    total_sequences = len(sequences)
    long_sequence_warnings = 0
    
    batches = batch_generator(sequences, batch_size)
    total_batches = (total_sequences + batch_size - 1) // batch_size
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Sequence', 'Prediction', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    for batch_idx, batch in enumerate(batches, start=1):
        batch_pids = []
        batch_seqs = []
        batch_cleaned_seqs = []
        
        for pid, seq in batch:
            seq_length = len(seq)
            if seq_length > 110:
                print(f"warning: sequence {pid} len: {seq_length},exceeding 110!,Prediction accuracy may be reduced for long sequences")
                long_sequence_warnings += 1
            
            cleaned_seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            batch_pids.append(pid)
            batch_seqs.append(seq)
            batch_cleaned_seqs.append(cleaned_seq)
        
        token_seqs = [' '.join(list(s)) for s in batch_cleaned_seqs]
        encoding = tokenizer.batch_encode_plus(
            token_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            embeddings = T5model(input_ids, attention_mask=attention_mask).last_hidden_state
            
            batch_embeddings = []
            for i, cleaned_seq in enumerate(batch_cleaned_seqs):
                seq_len = len(cleaned_seq)
                batch_embeddings.append(embeddings[i, :seq_len].unsqueeze(0))
            
            batch_outputs = []
            for embedding in batch_embeddings:
                output, _ = model(embedding)
                batch_outputs.append(output)
            
            
            batch_records = []
            for i, pid in enumerate(batch_pids):
                output = batch_outputs[i]
                pred = output.argmax(dim=1).cpu().numpy()[0]
                confidence = torch.softmax(output, dim=1)[:,1].detach().cpu().numpy()[0].round(5)
                
                batch_records.append({
                    "ID": pid,
                    "Sequence": batch_seqs[i],
                    "Prediction": pred,
                    "confidence": confidence
                })
        
            with open(output_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['ID', 'Sequence', 'Prediction', 'confidence'])
                writer.writerows(batch_records)
        
        print(f"Processed batch {batch_idx}/{total_batches} ({len(batch)}/{total_sequences} sequences)...")

    if long_sequence_warnings > 0:
        print(f"warning: {long_sequence_warnings} sequences exceeding len of 110! Prediction accuracy may be reduced for long sequences!")
    print(f"Results saved to {output_path}")   


def main():
    parser = argparse.ArgumentParser(description='Predict protein sequences from a FASTA file.')
    
    parser.add_argument('-i', '--input', help='Path to the input FASTA file.')
    parser.add_argument('-o', '--output', help='Path to save the output CSV file.')
    parser.add_argument('--model', default='./model/best_model.pth', help='Path to the trained model file.')
    parser.add_argument('--t5_model', default='./prot5uni50full', help='Path to the T5 model directory.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing (default: 8).')

    args = parser.parse_args()

    predict_from_fasta(args.input, args.model, args.t5_model, args.output, args.batch_size)

if __name__ == "__main__":
    main()

def save_attention_fasta(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        vocab: dict,
        save_path: str,
        threshold: float = 0.15,
        device: str = "cuda"
) -> None:
    model.eval()
    with open(save_path, "w") as f:

        for inputs, labels, seq, name in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs,attention_weights = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                if preds != labels:
                    continue
                attention = attention_weights
                #global_avg = attention.sum(axis=0)  # (seq_len,)
                global_avg =  attention.squeeze().cpu().numpy()
            # seq_indices = inputs.squeeze(0).cpu().numpy()
            # sequence = "".join([get_keys_by_value(vocab, i) for i in seq_indices])
            sequence = seq[0]
            seq_len = len(sequence)
            k = max(1, int(round(seq_len * threshold)))

            if threshold < 0.16:
                if k < 3:
                    k = 3
                if k > 10:
                    k = 10
            if threshold > 0.24:
                if k < 4:
                    k = 4
                if k > 20:
                    k = 20

            top_indices = np.argsort(global_avg)[-k:]
            projection = ["I"] * seq_len
            for idx in top_indices:
                projection[idx] = "S"
            f.write(f">{name[0]}|{labels.detach().cpu().numpy()[0]}\n")
            f.write(f"{sequence}\n")
            f.write(f"{''.join(projection)}\n")
