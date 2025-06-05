import ir_measures
import torch
import tqdm
import numpy as np
from ir_measures import Qrel, ScoredDoc, parse_measure, calc_aggregate

from gsasrec.gsasrec import GSASRec


def evaluate_pt(model: GSASRec, data_loader, metrics, limit, filter_rated, device):
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


def evaluate_onnx(session, data_loader, metrics, limit, filter_rated):
    processed_metrics = []
    for m in metrics:
        processed_metrics.append(parse_measure(m) if isinstance(m, str) else m)

    users_processed = 0
    scored_docs = []
    qrels = []

    input_name = session.get_inputs()[0].name
    num_items = session.get_outputs()[1].shape[0]

    for data, rated, target in tqdm.tqdm(data_loader, total=len(data_loader)):
        inputs_np = data.numpy().astype(np.int64)
        seq_emb_np, output_emb_np = session.run(None, {input_name: inputs_np})
        seq_emb = torch.from_numpy(seq_emb_np)
        output_emb = torch.from_numpy(output_emb_np)

        scores_full = torch.matmul(seq_emb, output_emb.t())
        scores_full[:, 0] = float("-inf")
        scores_full[:, num_items:] = float("-inf")

        if filter_rated:
            for i in range(len(data)):
                for j in rated[i]:
                    scores_full[i, j] = float("-inf")

        topk = torch.topk(scores_full, limit, dim=1)
        items, scores = topk.indices, topk.values

        for recommended_items, recommended_scores, target_item in zip(
            items, scores, target
        ):
            for item, score in zip(recommended_items, recommended_scores):
                scored_docs.append(
                    ScoredDoc(str(users_processed), str(int(item)), float(score))
                )
            qrels.append(Qrel(str(users_processed), str(int(target_item)), 1))
            users_processed += 1

    raw_result = calc_aggregate(processed_metrics, qrels, scored_docs)
    return {str(measure): value for measure, value in raw_result.items()}
