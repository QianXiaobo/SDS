#! /usr/local/bin/python
import gzip
from collections import defaultdict
import argparse
# import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="从 dosage 生成 SDS singletons, 分染色体运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--dosage", required=True, help="输入 gzipped dosage 文件")
    p.add_argument("--output", help="singletons output")
    p.add_argument("--nsamples", default=12201, help="样本量", type=int)
    # p.add_argument("--aa-tag", default="AA", help="VCF INFO 中祖先等位基因的标签 (default: AA)")
    # p.add_argument("--depth-file", default=None, help="个体平均深度文件（一行一个值，顺序同 VCF）")
    # p.add_argument("--min-daf", type=float, default=0.05, help="最小 DAF (default: 0.05)")
    # p.add_argument("--max-daf", type=float, default=0.95, help="最大 DAF (default: 0.95)")
    # p.add_argument("--min-geno-count", type=int, default=5, help="每种基因型的最小个体数 (default: 5)")
    # p.add_argument("--use-bcftools", action="store_true", help="不用 pysam，改用 bcftools 子进程")
    return p.parse_args()

def read_singleton(file1):
    with gzip.open(file1, 'rt') as fin:
        sample_variants = defaultdict(list)
        for line in fin:
        # for line in tqdm.tqdm(fin, desc="Processing variants", unit="lines"):
            arry = line.strip().split()
            row_idx = arry[0]
            for col_idx, val in enumerate(arry[1:]):
                if val == '1':
                    sample_variants[col_idx].append(row_idx)
        return(sample_variants)
    
def write_to_sds_input(var, output, n):
    with open(output, 'w') as fout:
        for x in range(0,n):
            if var[x] == []:
                fout.write("\n")
            else:
                fout.write(" ".join(var[x]))
                fout.write("\n")

if __name__ == "__main__":
    args = parse_args()
    dosage = args.dosage
    var = read_singleton(dosage)
    write_to_sds_input(var, args.output, args.nsamples)
    
    