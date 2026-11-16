import html
import re

import requests

URLS = [
    "https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html",
    "https://ojs.aaai.org/index.php/AAAI/article/view/25071",
    "https://doi.org/10.1007/s11263-024-02046-2",
    "https://openaccess.thecvf.com/content/CVPR2025/html/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.html",
    "https://proceedings.mlr.press/v306/li26db.html",
]
KEYS = [
    "citation_title",
    "citation_author",
    "citation_publication_date",
    "citation_conference_title",
    "citation_journal_title",
    "citation_volume",
    "citation_firstpage",
    "citation_lastpage",
    "citation_doi",
]


def meta_values(text: str, key: str) -> list[str]:
    pattern = rf'<meta[^>]+name=["\']{key}["\'][^>]+content=["\'](.*?)["\']'
    values = [html.unescape(item) for item in re.findall(pattern, text, re.I | re.S)]
    pattern = rf'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']{key}["\']'
    values += [html.unescape(item) for item in re.findall(pattern, text, re.I | re.S)]
    return values


for url in URLS:
    response = requests.get(url, timeout=30)
    print("\nURL", url, "status", response.status_code, "final", response.url)
    match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.I | re.S)
    if match:
        title = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        print("title", " ".join(title.split())[:220])
    for key in KEYS:
        values = meta_values(response.text, key)
        if values:
            print(key, values[:6])
