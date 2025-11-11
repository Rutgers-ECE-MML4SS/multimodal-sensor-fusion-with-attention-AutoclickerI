import json, os
def load(p): 
    with open(p) as f: return json.load(f)
root="experiments"
out=os.path.join(root,"fusion_comparison.json")
res={
  "dataset": "pamap2",
  "modalities": ["imu_hand","imu_chest","imu_ankle","heart_rate"],
  "results": {
    "early_fusion":  load(os.path.join(root,"early","evaluation_results.json")),
    "late_fusion":   load(os.path.join(root,"late","evaluation_results.json")),
    "hybrid_fusion": load(os.path.join(root,"hybrid","evaluation_results.json"))
  }
}
# 위 각 JSON에 있는 중복 필드 정리(필요시)
for k,v in res["results"].items():
    for kk in ["dataset","fusion_type"]:
        v.pop(kk, None)
with open(out,"w") as f: json.dump(res,f,indent=2)
print("wrote", out)