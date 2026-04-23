from docx import Document
import copy

doc_path = "/Users/mac/computerscience/15会计研究/manuscript/word/数据要素利用与资产定价效率v8.docx"
doc = Document(doc_path)

def replace_in_para(idx, old, new):
    p = doc.paragraphs[idx]
    full = p.text
    if old not in full:
        print(f"  [WARNING] [{idx}] NOT FOUND: '{old}'")
        return False
    # Replace in runs
    for run in p.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            print(f"  [{idx}] run-replaced: '{old}' -> '{new}'")
            return True
    # Fallback: concatenated replacement
    combined = ''.join(r.text for r in p.runs)
    if old in combined:
        # rebuild
        new_text = combined.replace(old, new)
        if p.runs:
            fmt = p.runs[0].font
            style_name = p.runs[0].style
            for r in p.runs[1:]:
                p._element.remove(r._element)
            p.runs[0].text = new_text
        print(f"  [{idx}] concat-replaced: '{old}' -> '{new}'")
        return True
    print(f"  [WARNING] [{idx}] Could not replace in runs: '{old}'")
    return False

print("=== 更新稳健性检验系数 ===")

# [80] FE robustness: -0.0022 -> -0.0028, -0.004 -> -0.0045
replace_in_para(80, "-0.0022", "-0.0028")
replace_in_para(80, "和-0.004", "和-0.0045")

# [81] Industry adj: -0.0020 -> -0.0025
replace_in_para(81, "-0.0020", "-0.0025")

# [82] Drop samples: IT coef -0.004 -> -0.0046, mainboard -0.003 -> -0.003 (stays as -0.003 rounded)
# Actually: noit=-0.0046, mainboard=-0.0032
# Current text: "-0.004（p<0.01），说明效应并非由IT行业主导" and "-0.003（p<0.05）"
replace_in_para(82, "系数为-0.004（p<0.01），说明效应并非由IT行业主导", 
                    "系数为-0.005（p<0.01），说明效应并非由IT行业主导")

# [83] PSM: N=34,425 -> 35,228, -0.004 -> -0.004 (rounds same at 3dp)
replace_in_para(83, "N=34,425", "N=35,228")
replace_in_para(83, "系数为-0.004", "系数为-0.004")  # stays same at 3dp

# [84] Two-way cluster: -0.004 -> -0.005
replace_in_para(84, "系数为-0.004（p<0.01），结论不受聚类方式选择的影响",
                    "系数为-0.005（p<0.01），结论不受聚类方式选择的影响")

# [85] Lead test: -0.005 -> -0.005 (stays same)
# 0.0005 stays same

# [86] IV section - multiple updates
print("\n=== 更新IV检验系数 ===")
# peer IV 2SLS: -0.0147 -> -0.0143
replace_in_para(86, "-0.0147", "-0.0143")
# peer IV ratio: 3.5倍 -> with -0.0143/0.005=2.9 ≈ 3倍
replace_in_para(86, "约为OLS的3.5倍", "约为OLS的3倍")
# Bartik F: 62.5 -> 60.4
replace_in_para(86, "F统计量为62.5", "F统计量为60.4")
# Bartik 2SLS: -0.0112（p=0.080），方向一致 -> -0.0105（p=0.173），方向一致但未通过显著性检验
replace_in_para(86, "二阶段系数为-0.0112（p=0.080），方向一致", 
                    "二阶段系数为-0.0105（p=0.173），方向一致但未通过显著性检验，可能与Bartik工具变量在本场景下的局部识别效率较低有关")
# Lag IV F: 183.5 -> 184.7
replace_in_para(86, "F统计量为183.5", "F统计量为184.7")
# Lag IV 2SLS: -0.0165 -> -0.0157
replace_in_para(86, "-0.0165", "-0.0157")
# Range update: (0.011至0.017) -> (0.011至0.016)
replace_in_para(86, "0.011至0.017", "0.011至0.016")

# [87] Oster: δ*=27.8 -> 76.3, R²max=0.553 -> recalculate
# Old: R²max=1.3R̃=0.553, δ*=27.8; conservative δ*=3.6
# New: R² full ≈ 0.419, so 1.3*0.419=0.545; δ*=76.3 both settings
print("\n=== 更新Oster系数稳定性 ===")
replace_in_para(87, "R²max=1.3R̃=0.553计算，δ*=27.8", "R²max=1.3R̃=0.545计算，δ*=76.3")
replace_in_para(87, "δ*=3.6", "δ*=76.3")

# [88] Lag OLS: -0.0036 -> -0.0039
print("\n=== 更新滞后回归系数 ===")
replace_in_para(88, "-0.0036", "-0.0039")

# Save
doc.save(doc_path)
print("\n=== Done! ===")

# Verify
doc2 = Document(doc_path)
for i in [80,81,82,83,84,86,87,88]:
    print(f"\n[{i}]: {doc2.paragraphs[i].text[:200]}")
