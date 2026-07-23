def disease_stats(infile,outfile):
    disease_count = {}
    with open(infile,'rt') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[2] not in disease_count:
                disease_count[items[2]] = [1,0]
            else:
                disease_count[items[2]][0] += 1
            if items[3] == '1':
                disease_count[items[2]][1] += 1
    with open(outfile,'wt') as out:
        for key, value in disease_count.items():
            out.write(f'{key}\t{value[0]}\t{value[1]}\t{value[1]/value[0]}\n')

infile = '90AA_human_secret.tsv'
outfile = 'disease_secret.tsv'
disease_stats(infile,outfile)


def filter_uniq(infile,outfile):
    with open(outfile,'wt') as out:
        with open(infile,'rt') as f:
            for line in f:
                linelist = line.strip().split('\t')
                disease = linelist[0].split('#')
                if len(disease) == 1:
                    out.write(line)


infile = 'disease_secret.tsv'
outfile = 'disease_secret_uniq.tsv'
filter_uniq(infile,outfile)

