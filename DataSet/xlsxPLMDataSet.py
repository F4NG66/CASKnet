import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
from torch.nn.utils.rnn import pad_sequence
def MySelf_collate_fullEmbed(batch):
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim] and metadate collated according to default collate

    """
    embeddings = [item[0] for item in batch]
    # localization = torch.tensor([item[1] for item in batch])
    lable = torch.tensor([item[1] for item in batch])
    sequence = [item[2] for item in batch]
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings, lable, sequence

class xlsxProteinDataset(Dataset):
    def __init__(self,csv_file, fasta_path, t5_file, max_length=110):
        """
        初始化数据集
        :param csv_file: 包含ID和可溶性的CSV文件
        :param fasta_file: 包含蛋白质序列的FASTA文件
        :param max_length: 最大序列长度，超过的会被截断
        """
        self.embeddings_file = h5py.File(t5_file, 'r')
        self.df = pd.read_csv(csv_file)
        # 读取FASTA文件并创建序列字典
        self.max_length = max_length
        self.num_classes = 2  # 二分类问题
        self.sequencesid = list()
        self.sequencesdict = dict()
        self.lenNameDict = dict()
        self.namemannalfeature = dict()
        self.MaxLength = 110

        with open(fasta_path, 'r') as fasta_f:
            for line in fasta_f:
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip()
                    uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                    self.sequencesid.append(uniprot_id)
                    self.sequencesdict[uniprot_id] = ''
                else:
                    self.sequencesdict[uniprot_id] += ''.join(line.split()).upper().replace("-","")
        for id in self.sequencesid:
            self.lenNameDict[id] = len(self.sequencesdict[id])


    def __len__(self):
        return len(self.sequencesid)

    def __getitem__(self, idx):
        protein_id = str(self.df.iloc[idx, 0])
        label = self.df.iloc[idx, 2]
        proteinseq = self.sequencesdict[protein_id]
        encoded = self.embeddings_file[protein_id][:]
        # seq = seq.upper().replace('U','X').replace('Z','X').replace('O','X') # 替换非常见氨基酸
        # 将序列转换为索引
        # seq_idx = [self.vocab.get(c, 0) for c in seq[:self.max_length]]

        return torch.tensor(encoded),torch.tensor(label,dtype=torch.long),proteinseq,protein_id

if  __name__ == '__main__':
    train_dataset = xlsxProteinDataset('./DataSetSoureFile/training_set.csv', './DataSetSoureFile/training_set.fasta','./DataProcess/ProEmbedding/training_set.h5')
    val_dataset  = xlsxProteinDataset('./DataSetSoureFile/test_set.csv', './DataSetSoureFile/test_set.fasta','./DataProcess/ProEmbedding/test_set.h5')
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn = MySelf_collate_fullEmbed)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,collate_fn = MySelf_collate_fullEmbed)