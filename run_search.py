import logging

from algodisco.providers.logger.pickle_logger import BasePickleLogger
from tasks.traveling_salesman.evaluator import Evaluator
from tasks.traveling_salesman.problem import task_description, template_program

from algodisco.providers.llm import OpenAIAPI
from algodisco.methods.funsearch_behavesim import BehaveSimSearchConfig, BehaveSimSearch
from algodisco.methods.funsearch_behavesim.similarity_calculator import (
    BehaveSimCalculator,
)
from algodisco.providers.logger.swanlab_logger import BaseSwanLabLogger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    llm = OpenAIAPI(
        model="gpt-5-nano",
        base_url="xxx",  # <--- Todo: Plz change to your base URL.
        api_key="xxx",  # <--- Todo: Plz change to your key.
    )
    evaluator = Evaluator(
        n_instance=5,
        problem_size=50,
    )

    # Todo: Uncomment following code if 'SwanLab' is not applicable for you !!!
    # Use normal pickle logger
    # logger = BasePickleLogger(logdir=f"logs/behavesim-search-run1")

    # Use SwanLab logger
    logger = BaseSwanLabLogger(
        logdir=f"logs/behavesim-search-run1",
        project="behavesim-tsp",
        experiment_name=f"behavesim-search-run1",
        group=f"behavesim-search",
        api_key="xxx",  # <--- Todo: Replace it with your api_key !!!
    )

    sim_calculator = BehaveSimCalculator(
        use_dtw=True,
        traj_sample_k=2,
        traj_trunc_n=20,
    )

    search_config = BehaveSimSearchConfig(
        task_description=task_description,
        template_program=template_program,
        num_samplers="auto",  # <--- Todo: Sampler parallel
        num_evaluators="auto",  # <--- Todo: Evaluation parallel (adjust based on the #CPU Cores)
        examples_per_prompt=2,
        samples_per_prompt=2,
        max_samples=2_000,
        db_save_frequency=500,
        llm_max_tokens=None,
        llm_timeout_seconds=60 * 3,
        enable_database_reclustering=True,
        recluster_threshold=100,
        db_algo_sim_calculator=sim_calculator,
    )

    search = BehaveSimSearch(
        config=search_config,
        llm=llm,
        evaluator=evaluator,
        logger=logger,
    )

    search.run()
    logging.info("FunSearch run completed.")


if __name__ == "__main__":
    main()
