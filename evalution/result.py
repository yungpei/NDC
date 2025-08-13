import json
import matplotlib.pyplot as plt
import os

# JSON files
json_files = {
    'pid_02': './pid_02/pid02_results.json',
    'pid_05': './pid_05/pid05_results.json',
    'pid_07': './pid_07/pid07_results.json',
    'pid_08': './pid_08/pid08_results.json',
    'pid_21': './pid_21/pid21_results.json'
}

# Metric mapping
metrics = {
    'ChamferL2': 'CD (L2)',
    'Hausdorff_max': 'Hausdorff Distance (max)',
    'F1_Score': 'F1-Score@τ=0.01',
    'NormalConsistency_avg': 'Normal Consistency'
}

# Prepare data container
data = {m: {'NDC': [], 'NMC': [], 'VTK': []} for m in metrics}

# Load JSON results
for pid, fname in json_files.items():
    with open(fname, 'r') as f:
        results = json.load(f)
    for m in metrics:
        data[m]['NDC'].append(results['NDC_OBJ'][m])
        data[m]['NMC'].append(results['NMC_OBJ'][m])
        data[m]['VTK'].append(results['VTK_STL'][m])

cases = list(json_files.keys())

# Create output directory
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

# Plot and save to PDF
for m_key, m_label in metrics.items():
    plt.figure(figsize=(6, 4))
    plt.plot(cases, data[m_key]['NDC'], marker='o', label='NDC')
    plt.plot(cases, data[m_key]['NMC'], marker='s', label='NMC')
    plt.plot(cases, data[m_key]['VTK'], marker='^', label='VTK')
    plt.title(f'{m_label} Across Cases')
    plt.xlabel('Case')
    plt.ylabel(m_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save as PDF
    pdf_path = os.path.join(output_dir, f'{m_key}_across_cases.pdf')
    plt.savefig(pdf_path)
    plt.close()