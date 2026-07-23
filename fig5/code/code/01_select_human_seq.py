def select(infile1,infile2,infile3,outfile):
    import lzma
    habitat_dict = {}
    with open(infile1,'rt') as f:
        for line in f:
            linelist = line.strip().split('\t')
            micro = linelist[2]+linelist[4]
            habitat_dict[micro] = linelist[5]

    sample_dict = {}
    with open(infile2,'rt') as f:
        for line in f:
            linelist = line.strip().split('\t')
            if len(linelist) > 23:
                if linelist[20] == '9606':
                    habitat = linelist[9]+linelist[20]
                    sample_dict[linelist[0]] = [habitat_dict[habitat],linelist[23]]

    with open(outfile,'wt') as out:
        with lzma.open(infile3,'rt') as f:
            for line in f:
                linelist = line.strip().split('\t')
                sample_list = linelist[1].split(',')
                env_set = set()
                disease_set = set()
                for sample in sample_list:
                    if sample in sample_dict.keys():
                        env_set.add(sample_dict[sample][0])
                        if sample_dict[sample][1] != '':
                            disease_set.add(sample_dict[sample][1])
                envs = '#'.join(sorted(list(env_set)))
                diseases = '#'.join(sorted(list(disease_set)))
                if envs != '':
                    out.write(f'{linelist[0]}\t{envs}\t{diseases}\n')
                            
infile1 = 'general_envo_names.tsv'
infile2 = 'metadata.tsv'
infile3 = '/data/yiqian/istbi/GMSC/final_frozen/GMSC10.90AA.sample.tsv.xz'
outfile = '90AA_human.tsv'
select(infile1,infile2,infile3,outfile)

def map_secret(infile1,infile2,outfile):
    with open(infile1,'rt') as f1:
        seq_dict = {}
        for line in f1:
            linelist = line.strip().split('\t',1)
            seq_dict[linelist[0]] = linelist[1]
    with open(outfile,'wt') as out:
        with open(infile2,'rt') as f2:
            for line in f2:
                linelist = line.strip().split(',')
                if linelist[0] in seq_dict.keys():
                    out.write(f'{seq_dict[linelist[0]]}\t{linelist[2]}\t{linelist[3]}')


infile1 = '90AA_human.tsv'
infile2 = '/data/ckzhu/yiqian/result_all.csv'
outfile = '90AA_human_secret.tsv'

def habitat_stats(infile,outfile):
    habitat_count = {}
    with open(infile,'rt') as f:
        for line in f:
            items = line.strip().split('\t')
            habitat_list = items[1].split('#')
            for item in habitat_list:
                if item not in habitat_count:
                    habitat_count[item] = [1,0]
                else:
                    habitat_count[item][0] += 1
                if items[3] == '1':
                    habitat_count[item][1] += 1
    with open(outfile,'wt') as out:
        for key, value in habitat_count.items():
            out.write(f'{key}\t{value[0]}\t{value[1]}\t{value[1]/value[0]}\n')

infile = '90AA_human_secret.tsv'
outfile = 'habitat_secret.tsv'
habitat_stats(infile,outfile)