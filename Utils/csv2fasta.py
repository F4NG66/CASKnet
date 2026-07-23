import pandas as pd

# 读取Excel文件
df = pd.read_csv("final_deduplicated_dataset.csv")
# df = pd.read_excel("final_deduplicated_dataset.csv",sheet_name="S2")
# 生成FASTA格式
fasta_lines = []
for _, row in df.iterrows():
    fasta_lines.append(f">{row['Entry']}\n{row['Sequence']}")

# 保存为.fasta文件
with open("secretepro.fasta", "w") as f:
    f.write("\n".join(fasta_lines))