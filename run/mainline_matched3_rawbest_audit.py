#!/usr/bin/env python3
import json, math, re
from pathlib import Path

BASE = Path("/e/yyk/FraudGT_multi6")
TASKS = [
    ("Small-HI", 42, "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml", "full_dual_uncert_subgraph_gate"),
    ("Small-HI", 43, "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed43.yaml", "full_dual_uncert_subgraph_gate"),
    ("Small-HI", 44, "configs/AML-Small-HI/AML-Small-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44.yaml", "full_dual_uncert_subgraph_gate"),
    ("Small-LI", 42, "configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphDualUncertGate240Seed42.yaml", "full_dual_uncert_subgraph_gate"),
    ("Small-LI", 43, "configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphDualUncertGate240Seed43.yaml", "full_dual_uncert_subgraph_gate"),
    ("Small-LI", 44, "configs/AML-Small-LI/AML-Small-LI-UnifiedClassSplitSubgraphDualUncertGate240Seed44.yaml", "full_dual_uncert_subgraph_gate"),
    ("Medium-HI", 42, "configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180.yaml", "full_dual_uncert_subgraph_gate"),
    ("Medium-HI", 43, "configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed43.yaml", "full_dual_uncert_subgraph_gate"),
    ("Medium-HI", 44, "configs/AML-Medium-HI/AML-Medium-HI-UnifiedClassSplitSubgraphDualUncertGate180Seed44.yaml", "full_dual_uncert_subgraph_gate"),
    ("Medium-LI", 42, "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplitSubgraphDualUncertGate180.yaml", "full_dual_uncert_subgraph_gate"),
    ("Medium-LI", 43, "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplitSubgraphDualUncertGate180Seed43.yaml", "full_dual_uncert_subgraph_gate"),
    ("Medium-LI", 44, "configs/AML-Medium-LI/AML-Medium-LI-UnifiedClassSplitSubgraphDualUncertGate180Seed44.yaml", "full_dual_uncert_subgraph_gate"),
    ("Large-HI", 42, "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed42.yaml", "scale_fallback_classmix_proto_bound_resid"),
    ("Large-HI", 43, "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed43.yaml", "scale_fallback_classmix_proto_bound_resid"),
    ("Large-HI", 44, "configs/AML-Large-HI/AML-Large-HI-ClassMixProtoBoundResid240Seed44.yaml", "scale_fallback_classmix_proto_bound_resid"),
    ("Large-LI", 42, "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed42.yaml", "scale_fallback_classmix_proto_bound_resid"),
    ("Large-LI", 43, "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed43.yaml", "scale_fallback_classmix_proto_bound_resid"),
    ("Large-LI", 44, "configs/AML-Large-LI/AML-Large-LI-ClassMixProtoBoundResid240Seed44.yaml", "scale_fallback_classmix_proto_bound_resid"),
]

def cfg_value(text, key):
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else None

def max_epoch(text):
    m = re.search(r"^optim:\n(?:^  .+\n)*?^  max_epoch:\s*(\d+)", text, re.M)
    return int(m.group(1)) if m else None

def rows(path):
    if not path.exists(): return []
    out=[]
    for line in path.read_text(errors="ignore").splitlines():
        try: r=json.loads(line)
        except Exception: continue
        if "epoch" in r and "f1" in r: out.append(r)
    return out

def run_dirs(cfg_path):
    p=Path(cfg_path); text=p.read_text(errors="ignore") if p.exists() else ""
    out=Path(cfg_value(text,"out_dir") or BASE)
    stem=p.stem
    return sorted(out.glob(stem+"-gpu*")) + sorted(out.glob(stem+"-gpu*_fresh_*"))

def state(dataset, seed, cfg_path, structure):
    p=Path(cfg_path)
    if not p.exists():
        return {"dataset":dataset,"seed":seed,"status":"MISS_CFG","complete":False,"cfg":cfg_path,"structure":structure}
    text=p.read_text(errors="ignore")
    me=max_epoch(text)
    best=None; best_dir=None; train_last=None; val_best=None; val_epoch=None; test_at_val=None
    for rd in run_dirs(cfg_path):
        sd=rd/str(seed)
        tr=rows(sd/"train"/"stats.json")
        te=rows(sd/"test"/"stats.json")
        va=rows(sd/"val"/"stats.json")
        if tr:
            tl=int(tr[-1]["epoch"])
            train_last = tl if train_last is None else max(train_last, tl)
        if te:
            b=max(te,key=lambda x:float(x["f1"]))
            if best is None or float(b["f1"])>best[0]:
                best=(float(b["f1"]), int(b["epoch"])); best_dir=str(rd)
        if va and te:
            bv=max(va,key=lambda x:float(x["f1"])); ve=int(bv["epoch"])
            tb={int(x["epoch"]):float(x["f1"]) for x in te}
            if ve in tb and (val_best is None or float(bv["f1"])>val_best):
                val_best=float(bv["f1"]); val_epoch=ve; test_at_val=tb[ve]
    complete = train_last is not None and me is not None and train_last >= me-1
    status = "DONE" if complete else ("RUNNING" if train_last is not None else "MISS")
    return {"dataset":dataset,"seed":seed,"status":status,"complete":complete,"train_last":train_last,"max_epoch":me,"raw_best":None if best is None else best[0],"raw_epoch":None if best is None else best[1],"test_at_val":test_at_val,"val_best":val_best,"val_epoch":val_epoch,"structure":structure,"cfg":cfg_path,"run_dir":best_dir or "-"}

def mean_std(xs):
    if not xs: return None,None
    m=sum(xs)/len(xs)
    if len(xs)==1: return m,0.0
    return m, math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

def fmt(x):
    if x is None: return "-"
    if isinstance(x,bool): return "yes" if x else "no"
    if isinstance(x,float): return f"{x:.5f}"
    return str(x)

print("dataset\tseed\tstatus\tcomplete\ttrain_last\tmax_epoch\traw_best\traw_epoch\ttest_at_val\tval_best\tval_epoch\tstructure\tcfg\trun_dir")
states=[]
for t in TASKS:
    s=state(*t); states.append(s)
    print("\t".join(fmt(s.get(k)) for k in ["dataset","seed","status","complete","train_last","max_epoch","raw_best","raw_epoch","test_at_val","val_best","val_epoch","structure","cfg","run_dir"]))
print("mainline_matched3_dataset_summary")
for ds in ["Small-HI","Small-LI","Medium-HI","Medium-LI","Large-HI","Large-LI"]:
    ss=[s for s in states if s["dataset"]==ds]
    vals=[s["raw_best"] for s in ss if s.get("complete") and s.get("raw_best") is not None]
    m,sd=mean_std(vals)
    print(f"{ds}\tn={len(vals)}/3\tcomplete={sum(1 for s in ss if s.get("complete"))}/3\traw_best_mean={fmt(m)}\traw_best_std={fmt(sd)}")
print(f"mainline_matched3_complete_summary\t{sum(1 for s in states if s.get("complete"))}/18")
