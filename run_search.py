import asyncio
import logging

from adtools.lm.openai_api import OpenAIAPI

from tasks.traveling_salesman.evaluator import Evaluator
from tasks.traveling_salesman.problem import task_description, template_program
from behavesim_search.search_async import BehaveSimSearchAsync, BehaveSimSearchConfig
from behavesim_search.similarity_calculator import BehaveSimCalculator
from behavesim_search.algo_database import AlgoDatabaseConfig
from behavesim_search.logger import PickleLoggerWithSwanLab, PickleLogger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def main():
    llm = OpenAIAPI(
        model="gpt-5-nano",
        base_url="xxx",  # <--- Todo: Plz change to your base URL.
        api_key="xxx",  # <--- Todo: Plz change to your key.
    )
    evaluator = Evaluator(
        n_instance=5,
        problem_size=50,
    )
    sim_calculator = BehaveSimCalculator(
        use_dtw=True,
        traj_sample_k=2,
        traj_trunc_n=20,
    )

    # Todo: Uncomment following code if SwanLab is not applicable for you !!!
    # Use normal pickle logger
    # logger = PickleLogger(logdir=f"logs/behavesim-search-run1")

    # Use SwanLab logger
    logger = PickleLoggerWithSwanLab(
        logdir=f"logs/behavesim-search-run1",
        project="behavesim-tsp",
        experiment_name=f"behavesim-search-run1",
        group=f"behavesim-search",
        api_key="xxx",  # <--- Todo: Replace it with your api_key !!!
    )

    db_config = AlgoDatabaseConfig(
        algo_sim_calculator=sim_calculator,
        n_islands=10,
        num_sim_caculator_workers=2,
        async_register=True,
    )

    search_config = BehaveSimSearchConfig(
        task_description=task_description,
        database_config=db_config,
        template_program=template_program,
        evaluate_timeout_seconds=60.0,
        num_samplers=8,  # <--- Todo: Sampler parallel
        num_evaluators=4,  # <--- Todo: Evaluation parallel (adjust based on the #CPU Cores)
        examples_per_prompt=2,
        samples_per_prompt=2,
        max_samples=2_000,
        db_save_frequency=500,
        redirect_to_devnull=True,
        llm_max_tokens=None,
        llm_timeout_seconds=60 * 3,
        enable_database_reclustering=True,
        recluster_threshold=100,
    )

    search = BehaveSimSearchAsync(
        config=search_config,
        llm=llm,
        evaluator=evaluator,
        logger=logger,
    )

    await search.run()
    logging.info("FunSearch run completed.")


if __name__ == "__main__":
    asyncio.run(main())
