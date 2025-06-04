import ir_measures
import torch
import tqdm
from ir_measures import Qrel, ScoredDoc, parse_measure

from gsasrec.gsasrec import GSASRec


def evaluate(model: GSASRec, data_loader, metrics, limit, filter_rated, device):
    model.eval()

    processed_metrics = []
    for m in metrics:
        if isinstance(m, str):
            processed_metrics.append(parse_measure(m))
        else:
            processed_metrics.append(m)

    users_processed = 0
    scored_docs = []
    qrels = []
    with torch.no_grad():
        max_batches = len(data_loader)
        for batch_idx, (data, rated, target) in tqdm.tqdm(
            enumerate(data_loader), total=max_batches
        ):
            data, target = data.to(device), target.to(device)
            if filter_rated:
                items, scores = model.get_predictions(data, limit, rated)
            else:
                items, scores = model.get_predictions(data, limit)
            for recommended_items, recommended_scores, target_item in zip(
                items, scores, target
            ):
                for item, score in zip(recommended_items, recommended_scores):
                    scored_docs.append(
                        ScoredDoc(str(users_processed), str(item.item()), score.item())
                    )
                qrels.append(Qrel(str(users_processed), str(target_item.item()), 1))
                users_processed += 1

    raw_result = ir_measures.calc_aggregate(processed_metrics, qrels, scored_docs)

    string_keyed = {str(measure): value for measure, value in raw_result.items()}
    return string_keyed
