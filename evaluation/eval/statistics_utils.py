def compute_statistics(eval_result, mode, output_dir, show=False):
    """Statistics summary for different evaluation modes."""
    stats = {}
    if mode == "instruction-compliance":
        op_counts, op_success = {}, {}
        total_count, total_success = 0, 0
        for k, v in eval_result.items():
            op = v["operation"]
            op_counts[op] = op_counts.get(op, 0) + 1
            op_success[op] = op_success.get(op, 0) + v["compliance"]
            total_success += v["compliance"]
            total_count += 1
        stats["operation_success_rate"] = {op: {"rate": op_success.get(op, 0) / op_counts[op], "count": op_counts[op]} for op in op_counts}
        stats["overall_success_rate"] = total_success / total_count if total_count > 0 else 0.0
        stats["overall_total_count"] = total_count
        if show:
            print("Instruction-compliance success rate per operation:")
            for op, rate_dict in stats["operation_success_rate"].items():
                rate = rate_dict.get("rate", 0.0)
                count = rate_dict.get("count", 0)
                print(f"  {op}: rate={rate:.4f}, count={count}")
            print(f"Overall success rate: {stats['overall_success_rate']:.4f}")

    elif mode == "spatial-accuracy":
        op_scores, op_counts = {}, {}
        total_score, total_count = 0.0, 0
        for k, v in eval_result.items():
            op = v["operation"]
            op_scores[op] = op_scores.get(op, 0.0) + v["edit_score"]
            op_counts[op] = op_counts.get(op, 0) + 1
            total_score += v["edit_score"]
            total_count += 1
        stats["operation_mean_score"] = {op: op_scores[op] / op_counts[op] for op in op_scores}
        stats["overall_mean_score"] = total_score / total_count if total_count > 0 else 0.0
        if show:
            print("Spatial-accuracy mean edit_score per operation:")
            for op, mean_score in stats["operation_mean_score"].items():
                print(f"  {op}: {mean_score:.4f}")
            print(f"Overall mean edit_score: {stats['overall_mean_score']:.4f}")

    elif mode == "edit-locality":
        op_ssim, op_lpips, op_mse, op_counts = {}, {}, {}, {}
        total_ssim = 0.0
        total_lpips = 0.0
        total_mse = 0.0
        total_count = 0
        for k, v in eval_result.items():
            op = v.get("operation", "all")
            op_ssim[op] = op_ssim.get(op, 0.0) + v["ssim"]
            op_lpips[op] = op_lpips.get(op, 0.0) + v["lpips"]
            op_mse[op] = op_mse.get(op, 0.0) + v["mse"]
            op_counts[op] = op_counts.get(op, 0) + 1
            total_ssim += v["ssim"]
            total_lpips += v["lpips"]
            total_mse += v["mse"]
            total_count += 1
        stats["operation_mean_ssim"] = {op: op_ssim[op] / op_counts[op] for op in op_ssim}
        stats["operation_mean_lpips"] = {op: op_lpips[op] / op_counts[op] for op in op_lpips}
        stats["overall_mean_ssim"] = total_ssim / total_count if total_count > 0 else 0.0
        stats["overall_mean_lpips"] = total_lpips / total_count if total_count > 0 else 0.0
        if show:
            print("Edit-locality mean SSIM/LPIPS per operation:")
            for op in op_ssim:
                print(f"  {op}: SSIM={stats['operation_mean_ssim'][op]:.4f}, LPIPS={stats['operation_mean_lpips'][op]:.4f}")
            print(f"Overall mean SSIM: {stats['overall_mean_ssim']:.4f}")
            print(f"Overall mean LPIPS: {stats['overall_mean_lpips']:.4f}")
    
    return stats