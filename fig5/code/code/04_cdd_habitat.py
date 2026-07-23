def cal_sample(infile1,infile2,infile3,outfile):
    import lzma
    samples= set()
    with open(infile1,'rt') as f:
        for line in f:
            linelist = line.strip().split('\t')
            if linelist[23] == 'CDI':
                samples.add(linelist[0])

    with open(infile2,'rt') as f:
        secret_set = set()
        for line in f:
            linelist = line.strip().split('\t')
            if linelist[2] == 'CDI' and linelist[3] == '1':
                secret_set.add(linelist[0])

    with lzma.open(infile3,'rt') as f:
        with open(outfile,'wt') as out:
            seq_dict = {}
            for line in f:
                linelist = line.strip().split('\t')
                if linelist[0] in secret_set:
                    seq_dict[linelist[0]] = 0
                    sample_list = linelist[1].split(',')
                    for item in sample_list:
                        if item in samples:
                            seq_dict[linelist[0]] += 1
            for key,value in seq_dict.items():
                out.write(f'{key}\t{value}\n')
               

infile1 = '/data/yiqian/istbi/GMSC/secret/metadata_CDI.tsv'
infile2 = '/data/yiqian/istbi/GMSC/secret/90AA_human_CDI_secret.tsv'
infile3 = '/data/yiqian/istbi/GMSC/final_frozen/GMSC10.90AA.sample.tsv.xz'
outfile = '/data/yiqian/istbi/GMSC/secret/cdi/cdi_sample.tsv'
cal_sample(infile1,infile2,infile3,outfile)

def cal_habitat(infile1,outfile):
    habitat_dict = {}
    family_dict = {}
    with open(infile1,'rt') as f:
        for line in f:
            linelist = line.strip().split('\t')
            if linelist[0] not in family_dict.keys():
                family_dict[linelist[0]] = 1
            else:
                family_dict[linelist[0]] += 1
            habitat_list = linelist[5].split(',')
            for habitat in habitat_list:
                if habitat.startswith('human'):
                    habitat = habitat.replace('"','')
                    family_habitat = f'{linelist[0]}\t{habitat}'
                    if family_habitat not in habitat_dict.keys():
                        habitat_dict[family_habitat] = 1
                    else:
                        habitat_dict[family_habitat] += 1

    with open(outfile,'wt') as out:
        for key,value in habitat_dict.items():
            family = key.split('\t')[0]
            proportion = value/family_dict[family]
            out.write(f'{key}\t{value}\t{family_dict[family]}\t{proportion}\n')

infile1 = 'cdi_90AA.txt'
outfile = 'cdi_90AA_habitat.tsv'
cal_habitat(infile1,outfile)

def cal_tax(infile1,outfile):
    tax_dict = {}
    family_dict = {}
    with open(infile1,'rt') as f:
        for line in f:
            linelist = line.strip().split('\t')
            if linelist[0] not in family_dict.keys():
                family_dict[linelist[0]] = 1
            else:
                family_dict[linelist[0]] += 1
                
            tax_list = linelist[6].split(';')
            if len(tax_list) >5:
                family_tax = f'{linelist[0]}\t{tax_list[5]}'
                if family_tax not in tax_dict.keys():
                    tax_dict[family_tax] = 1

    with open(outfile,'wt') as out:
        for key,value in tax_dict.items():
            out.write(f'{key}\t{value}\n')

infile1 = 'cdi_90AA.txt'
outfile = 'cdi_90AA_taxonomy.tsv'
cal_tax(infile1,outfile)