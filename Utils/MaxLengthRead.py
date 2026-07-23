# 把多行的蛋白质转换为单行
from Bio import SeqIO

def ProteinMaxLength(input_file):
    MaxLength = 0
    with open(input_file, 'r') as infile:
        # 逐条读取 Fasta 文件中的记录
        for record in SeqIO.parse(infile, "fasta"):
            # 合并序列为一行，去除任何换行符
            sequence = str(record.seq).replace("\n", "")  # 这里确保序列是连贯的
            if len(sequence)>MaxLength:
                MaxLength = len(sequence)
    print("MaxLength is ", MaxLength)




if __name__ == '__main__':
    # 示例用法
    # input_file = "./Protein_dataset/test_dataset.fasta"  # 输入文件名
    # output_file = "./MergeDataSet/test_dataset.fasta"  # 输出文件名
    input_file1 = "../secretepro.fasta"  # 输入文件名
    # input_file2= "./test_dataset.fasta"  # 输入文件名
    # input_file3 = "./validation_dataset.fasta"  # 输入文件名
    ProteinMaxLength(input_file1)
    # ProteinMaxLength(input_file2)
    # ProteinMaxLength(input_file3)
