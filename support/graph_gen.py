import os
import numpy as np
import re

def counter_gen(out_file, tealeaf_in, jacobi_in, stats ):
    dataset = {}
    
    verify_matcher = re.compile("Cusan runtime statistics")
    cusan_matcher = re.compile(r".*_calls\s*:\s*\d+")
    tsan_matcher = re.compile(r"Tsan.*\s*:\s*\d+")
    for filename in [tealeaf_in, jacobi_in]:
        print(filename)
        with open(filename) as file:
            data = file.read()
            if (len(verify_matcher.findall(data))==0):
                print(f"ERROR: Couldnt get CuSan runtime stats from {filename}")
                continue

        res = cusan_matcher.findall(data) + tsan_matcher.findall(data)
        res = [[x.split(":")[0].strip(), int(x.split(":")[1])] for x in res]
        exp_name = "-".join(filename.split("/")[-1:]).split(".")[0]
        if exp_name not in dataset:
            dataset[exp_name] = {}
        dataset[exp_name] = dict(res) 



    format = "c|c|c"

    title = "Metric & " + " & ".join([x for x in dataset.keys()])

    datas = ""

    for x in stats:
        if x in dataset[list(dataset.keys())[0]]:

            datas += f"{x} & " + " & ".join([str(dataset[k][x]) for k in dataset.keys()]) + "\\\\\n"
        else:
            datas += x


    output = f"""\\begin{{tabular}}{{{format}}}\n{title}\\\n{datas}\\end{{tabular}}
    """
    output = r"\rowcolors{2}{gray!25}{white}" + output
    print(output)



def scale_gen(out_file, folder):
    dataset = {}
    matcher_mtime = re.compile(r"MTime=([.0-9]*)")
    matcher_maxrss = re.compile(r"MAX RSS\[[^\]]*\] during execution: ([0-9]*)")
    for filename in os.listdir(folder):
        if filename.endswith(".out"):
            continue
        mtimes = []
        maxrsss = []

        with open(folder + filename) as file:
            data = file.read()
            mtime = [float(x) for x in matcher_mtime.findall(data)]
            mtime.remove(max(mtime))

            maxrss = [int(x) for x in matcher_maxrss.findall(data)]
            maxrss = [x+y for (x,y) in zip(maxrss[::2], maxrss[1::2])]
            maxrss.remove(max(maxrss))

            mtimes.append(mtime)
            maxrsss.append(maxrss)

        file = filename.replace("256", "0256").replace("512", "0512")
        exp_name = "-".join(file.split("-")[:-1])
        if exp_name not in dataset:
            dataset[exp_name] = {}
        dataset[exp_name][int(file.split("-")[-1].split(".")[0])] = (mtimes, maxrsss) 

    for off in range(2):
        name = [
            "Runtime",
            "Max RSS(KB)"
        ]
        maxi = np.max([np.max([np.max(y[1][off]) for y in x.items()]) for x in dataset.values()])
        print("name", name[off])
        print("  ymax", maxi+maxi*0.2)
        styles = [
           "color=black, line width=1.5pt, mark size=2.3pt, mark=triangle, mark options={solid, black}" ,
           "color=white!40!black, line width=1.5pt, mark size=2.3pt, mark=triangle, mark options={solid, white!40!black}",
           "color=white!40!black, dashed, line width=1.5pt, mark size=3.5pt, mark=o, mark options={solid, white!40!black}",
           "color=black, dashed, line width=1.5pt, mark size=3.5pt, mark=o, mark options={solid, black}"
        ]
        i = 0
    
        for f, d in dataset.items():
            sorted_data = sorted(d.items(), key = lambda x: x[0])
            sorted_values = [np.mean(d[1][off]) for d in sorted_data]
            print("  ", f)
            print("    ", styles[i])
            print("     256: ",str(sorted_values[0]))
            print("     512: ",str(sorted_values[1]))
            print("     1024: ",str(sorted_values[2]))
            print("     2048: ",str(sorted_values[3]))
            print("     4096: ",str(sorted_values[4]))
            i += 1

def mem_runtime_getdataset(folder):
    dataset = {}
    
    matcher_mtime = re.compile(r"MTime=([.0-9]*)")
    matcher_maxrss = re.compile(r"MAX RSS\[[^\]]*\] during execution: ([0-9]*)")
    for filename in os.listdir(folder):
        if filename.endswith(".out"):
            continue
        mtimes = []
        maxrsss = []
        print(filename)
        with open(folder + filename) as file:
            data = file.read()
            mtime = [float(x) for x in matcher_mtime.findall(data)]
            mtime.remove(max(mtime))

            maxrss = [int(x) for x in matcher_maxrss.findall(data)]
            maxrss = [x+y for (x,y) in zip(maxrss[::2], maxrss[1::2])]
            maxrss.remove(max(maxrss))

            mtimes.append(mtime)
            maxrsss.append(maxrss)
        # exp_name = filename.split("-")[1].split(".")[0]
        exp_name = "-".join(filename.split("-")[1:]).split(".")[0]
        if exp_name not in dataset:
            dataset[exp_name] = {}
        dataset[exp_name] = (np.mean(mtimes), np.mean(maxrsss)) 

    vanilla = [dataset["vanilla"][0], dataset["vanilla"][1]]
    del dataset["vanilla"]
    for d in dataset:
        dataset[d] = (dataset[d][0]/vanilla[0],
                        dataset[d][1]/vanilla[1])
    return dataset


def mem_runtime_handle(out_name, folder1, folder2):

    dataset1 = mem_runtime_getdataset(folder1)
    dataset2 = mem_runtime_getdataset(folder2)


    for off in range(2):
        maxi = max(max([v[off] for v in dataset1.values()]), max([v[off] for v in dataset2.values()]))
        mini = min(min([v[off] for v in dataset1.values()]), min([v[off] for v in dataset2.values()]))

        titles = [
            "ylabel={{{\\scriptsize Rel. runtime} $\\left[\\frac{T_{\\text{Flavor}}}{T_{\\text{Vanilla}}}\\right]$}},",
            "ylabel={{{\\scriptsize Rel. memory} $\\left[\\frac{M_{\\text{Flavor}}}{M_{\\text{Vanilla}}}\\right]$}},",
        ]

        print("Title", titles[off])
        print("  ymax", maxi+maxi*0.2) 
        styles = [
            "ybar, bar width=0.2, fill=c1, draw=black, area legend",
            "ybar, bar width=0.2, fill=c2, draw=black, area legend",
            "ybar, bar width=0.2, fill=c3, draw=black, area legend",
            "ybar, bar width=0.2, fill=c4, draw=black, area legend",
            "ybar, bar width=0.2, fill=c5, draw=black, area legend",
            # "ybar, bar width=0.2, fill=c6, draw=black, area legend",
            # "ybar, bar width=0.2, fill=c7, draw=black, area legend",
        ]

        i = 0
        keys = set(dataset1.keys()) & set(dataset2.keys())
        order = [
            "vanilla",
            "vanilla-tsan",
            "vanilla-must-tsan",
            "cusan",
            "must-cusan",
        ]
        for key in sorted(list(keys), key = lambda x: order.index(x)):
            v1 = dataset1[key]
            v2 = dataset2[key]
            print("  ", key)
            print("     2.4", v1[off])
            print("     1", v2[off])
            i += 1




if __name__ == "__main__":
    #generates table of counters given outputname an 2 files and a list of stats
    print("==== COUNTER ====")
    counter_gen("./out/counter",
                   "./data/tealeaf-counter-46158893/tealeaf-cusan.txt",
                   "./data/jacobi-counter-46158891/jacobi-cusan.txt",
                    ["\\hline\\\\\n", "kernel_register_calls", "TsanMemoryRead", "TsanMemoryWrite", "TsanSwitchToFiber"]
                    )
    #generates scaling tikz graph based on outname and folder of bench files
    print("==== SCALE ====")
    scale_gen("./out/jacobi-scale", "./data/jacobi-scale-46158883/")

    #generates mem/runtime tikz graph based on outname and 2 folders of bench files
    print("==== MEM/RUNTIME ====")
    mem_runtime_handle("./out/jacobi", "./data/jacobi-46158882/", "./data/tealeaf-46158881/")



