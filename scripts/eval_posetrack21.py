"""
TODO
"""
import posetrack21.api as pt21_eval_api

from dgs.utils.files import to_abspath

if __name__ == "__main__":
    eval_type = "tracking"
    evaluator = pt21_eval_api.get_api(
        trackers_folder=to_abspath("./data/PoseTrack21/...predictions..."),
        gt_folder=to_abspath("./data/PoseTrack21/posetrack_data/annotations/val/"),
        eval_type=eval_type,
        num_parallel_cores=8,
        use_parallel=True,
    )

    # Obtain results for each evaluation threshold and for each joint class respectively, i.e. 19x16.
    # The last element, i.e. results['HOTA'][:, -1] is the total score over all key points.
    results = evaluator.eval()

    avg_results = evaluator.get_avg_results(results)

    print(f"Average results: {avg_results}")
