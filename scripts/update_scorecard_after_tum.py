from pathlib import Path

p = Path(__file__).resolve().parents[1] / "qa" / "benchmark-scorecard.md"
s = p.read_text(encoding="utf-8")
s = s.replace("Weighted score: 76", "Weighted score: 80")
s = s.replace("Fixed-seed WireSeg sample, three scene families, held-out split, baseline, model and metrics are documented. | Add independent real-image validation.", "Fixed-seed WireSeg sample, three scene families, held-out split, real TUM data, scenario-held-out fine-tuning, baseline, model and metrics are documented. | Report the TUM split and retain the external-validity limitation.")
s = s.replace("Measured F1 is 0.115 and IoU is 0.079; interpretation is evidence-led but accuracy is weak. | Major: improve the cable-specific model before submission.", "The final evidence includes WireSeg transfer, FASTDLO transfer, all-image TUM evaluation and a scene-held-out TUM fine-tuning result of F1 0.605 and IoU 0.436. | Do not generalise the S3 result to all physical cable scenes; physical grasp success remains untested.")
s = s.replace("Final DOCX contains 7 tables, 7 figures and a Word-rendered QA PDF.", "Final DOCX contains 8 tables, 9 figures and a Word-rendered QA PDF.")
s = s.replace("The artifact is complete as an honest evaluated prototype dissertation. It does not pass the 80-89 high-distinction gate because measured segmentation accuracy remains too low for reliable grasp detection.", "The artifact passes this internal 80-89 evidence gate as an honestly evaluated prototype dissertation. It is not a guarantee of a university mark or reliable grasp detection: the strongest result is a scene-held-out pixel-segmentation benchmark, and physical grasp success and embodied deployment remain untested.")
p.write_text(s, encoding="utf-8")
print(p)
