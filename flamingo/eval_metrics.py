import json
from eval_datasets import CaptionDataset
from aac_metrics.functional import cider_d, bleu_4, rouge_l
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents

def get_metrics_captioning(res_file, annotations_path, image_dir_path):
    with open(res_file) as f:
        data = json.load(f)
    
    model_outputs = {}
    for k,item in data["outputs"].items():
        model_outputs[k] = item[len(data["query"]):]
    
    dataset = CaptionDataset(image_dir_path, annotations_path)
    hyps, refs = [], []
    for k,v in model_outputs.items():
        hyps.append(v.replace('\n',' '))
        refs.append(dataset.data_dict[int(k)]["captions"])

    candidates = preprocess_mono_sents(hyps)
    mult_references = preprocess_mult_sents(refs)
    
    cider_scores, _ = cider_d(candidates, mult_references)
    bleu_scores, _  = bleu_4(candidates, mult_references)
    rouge_scores, _  = rouge_l(candidates, mult_references)
    print("Corpus BLEU Score:", bleu_scores)
    print("Corpus ROUGE Scores:", rouge_scores)
    print("Corpus Cider Scores:", cider_scores)
    return {
        "bleu": bleu_scores,
        "rouge": rouge_scores,
        "cider": cider_scores
    }