def map_cdd(infile1,infile2,outfile1):
    import lzma
    cdd_dict = {}
    with lzma.open(infile1,'rt') as f:
        for line in f:
            family,cdd = line.strip().split('\t')
            cdd_dict[family] = cdd
    with open(outfile1,'wt') as out:
        with open(infile2,'rt') as f:
            for line in f:
                linelist = line.strip().split('\t')
                if linelist[3] == '1':
                    if linelist[0] in cdd_dict.keys():
                        out.write(f'{line.strip()}\t{cdd_dict[linelist[0]]}\n')
                    else:
                        out.write(f'{line.strip()}\tNA\n')

infile1 = '/data/yiqian/istbi/GMSC/final_frozen/GMSC10.90AA.cdd.tsv.xz'
infile2 = '/data/yiqian/istbi/GMSC/secret/90AA_human_CDI_secret.tsv'
outfile1 = '/data/yiqian/istbi/GMSC/secret/cdi/cdi_cdd.tsv'
map_cdd(infile1,infile2,outfile1)

def map_cog(infile1,infile2,infile3,outfile):
    cdd_dict = {}
    with open(infile1,'rt') as f:
        for line in f:
            linelist = line.strip().split('\t')
            cdd_dict[linelist[0]] = linelist[1]
    cog_dict = {}
    with open(infile2,'rt') as f:
        for line in f:
            linelist = line.strip().split('\t')
            cog_dict[linelist[0]] = linelist[1]
    with open(outfile,'wt') as out:
        with open(infile3,'rt') as f:
            for line in f:
                linelist = line.strip().split('\t')
                cdd_list = linelist[5].split(',')
                final_cdd_list = []
                for item in cdd_list:
                    if item in cdd_dict.keys():
                        final_cdd_list.append(cdd_dict[item])
                final_cdd = ','.join(final_cdd_list) if len(final_cdd_list) > 0 else 'NA'
                if final_cdd != 'NA':
                    out.write(f'{line.strip()}\t{final_cdd}\n')

infile1 = 'cddid_all.tbl'
infile2 = 'cog-20.def.tab.tsv'
infile3 = 'cdi_cdd.tsv'
outfile = 'cdi_cdd_cog.tsv'
map_cog(infile1,infile2,infile3,outfile)

def cal_pfam(infile,outfile):
    with open(infile,'rt') as f:
        cdi_pfam_count = {}
        ctr_pfam_count = {}
        cdi = 0
        ctr = 0
        for line in f:
            linelist = line.strip().split('\t')
            pfam_list = linelist[6].split(',')
            for item in pfam_list:
                if item.startswith('pfam'):
                    if linelist[2] == 'CDI':
                        cdi += 1
                        if item in cdi_pfam_count.keys():
                            cdi_pfam_count[item] += 1
                        else:
                            cdi_pfam_count[item] = 1
                    elif linelist[2] == 'CTR':
                        ctr += 1
                        if item in ctr_pfam_count.keys():
                            ctr_pfam_count[item] += 1
                        else:
                            ctr_pfam_count[item] = 1
    with open(outfile,'wt') as out:
        out.write('Pfam_ID\tCDI_count\tCDI_all\tCTR_count\tCTR_all\n')
        for key in cdi_pfam_count.keys():
            if key in ctr_pfam_count.keys():
                out.write(f'{key}\t{cdi_pfam_count[key]}\t{cdi}\t{ctr_pfam_count[key]}\t{ctr}\n')
            else:
                out.write(f'{key}\t{cdi_pfam_count[key]}\t{cdi}\t0\t{ctr}\n')
        for key in ctr_pfam_count.keys():
            if key not in cdi_pfam_count.keys():
                out.write(f'{key}\t0\t{cdi}\t{ctr_pfam_count[key]}\t{ctr}\n')


infile  = 'cdi_cdd_cog.tsv'
outfile = 'cdi_pfam_count.tsv'
cal_pfam(infile,outfile)