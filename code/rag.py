from react import MyRAG
import json
import os
import csv
import pickle
from metadata_search_functions import finish_agent
from llama_index.core import QueryBundle
from react import CONTEXT_AGENT
from llama_index.core.agent import ReActAgent
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import uuid
import io
import contextlib
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_question(args):
    question, myrag, data, num_attempts, mode = args
    agent = ReActAgent.from_tools(
        myrag.tools,
        llm=myrag.llm,
        # verbose=True,
        max_iterations=10,
        contexts=CONTEXT_AGENT,
    )
    
    if mode == "ans":
        response = {
            "question": question,
            "answer": None,
            "ground_truth": data["ground_truth"],
            "citations": data["citations"],
            "type": data["type"]
        }
    else:  # unans mode
        response = {
            "question": question,
            "answer": None
        }

    for attempt in range(num_attempts):
        try:
            logger.info(f">>>>>>>>> QUESTION >>>>>>>>>>> {question}")
            logger.info(f">>>>>>>>> ATTEMPT >>>>>>>>>>> {attempt}")
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                answer = agent.chat(question)
            logger.info(f">>>>>>>>> ANSWER >>>>>>>>>>> {answer}")
            done = finish_agent(question, answer)
            if done:
                response["answer"] = answer
                return response
            
            # Reset agent if attempt failed
            agent = ReActAgent.from_tools(
                myrag.tools,
                llm=myrag.llm,
                # verbose=True,
                max_iterations=10,
                contexts=CONTEXT_AGENT,
            )
            
        except Exception as e:
            print(f"Error processing question with ReAct agent: {e}")
            continue
    
    try:
        fallback_answer = myrag.fallback_answer(question)
        if fallback_answer:
            response["answer"] = fallback_answer
    except Exception as e:
        print(f"Error processing question with fallback agent: {e}")
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["ans", "unans"],
        default="ans",
        help="Mode to run in: 'ans' for answerable questions, 'unans' for unanswerable questions",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="text-embedding-3-large",
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--embedding_mode",
        choices=["openai", "huggingface"],
        default="openai",
        help="Embedding mode to use (openai or huggingface)",
    )
    parser.add_argument(
        "--chunk_size",
        default=256,
        type=int,
        help="Size of text chunks for processing",
    )
    parser.add_argument(
        "--chunk_overlap",
        default=32,
        type=int,
        help="Overlap between consecutive chunks",
    )
    parser.add_argument(
        "--max_workers",
        default=5,
        type=int,
        help="Number of parallel workers to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o"
        # gemini-2.5-flash
    )   # deepseek-ai/DeepSeek-V3 meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 Qwen/Qwen3-235B-A22B-fp8-tput meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo deepseek-ai/DeepSeek-R1 Qwen/Qwen2.5-72B-Instruct-Turbo Qwen/QwQ-32B meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
    parser.add_argument(
        "--llm_provider",
        choices=["openai", "togetherai", "gemini"],
        default="openai",
        help="LLM provider to use (openai or togetherai)",
    )
    parser.add_argument(
        "--num_attempts",
        default=1,
        type=int,
        help="Number of attempts to make for each question",
    )
    parser.add_argument(
        "--reuse_index",
        action="store_true",
        help="Reuse existing indices if they exist",
    )
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    product_dir = "../data/products"
    save_index_path = f"index/{args.embedding_model_name}/all"
    load_index_path = save_index_path if args.reuse_index else None

    myrag = MyRAG(
        product_data=product_dir,
        load_index_path=load_index_path,
        save_index_path=save_index_path,
        embedding_mode=args.embedding_mode,
        embedding_model_name=args.embedding_model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        llm_provider=args.llm_provider
    )

    product_files = glob.glob(os.path.join(product_dir, "*.json"))
    qa_dataset = []

    for product_file in product_files:
        with open(product_file, 'r') as f:
            product_data = json.load(f)
            if args.mode == "unans" and "unanswerable_questions" in product_data:
                qa_dataset.extend([{"question": q} for q in product_data["unanswerable_questions"]])
            elif args.mode == "ans" and "answerable_questions" in product_data:
                qa_dataset.extend(product_data["answerable_questions"])

    if args.mode == "ans":
        dataset = {
            "answer": [],
            "ground_truth": [],
            "question": [],
            "citations": [],
            "type": []
        }
    else:  # unans mode
        dataset = {
            "answer": [],
            "question": []
        }

    start_time = time.time()
    processed_count = 0
    
    output_file = f'output/{args.embedding_model_name}-{args.model_name.replace("/", "_")}-{args.num_attempts}-{args.mode}.pk'

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        tasks = [(data["question"], myrag, data, args.num_attempts, args.mode) for data in qa_dataset]

        futures = [executor.submit(process_question, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
            result = future.result()
            if result:
                dataset["question"].append(result["question"])
                dataset["answer"].append(result["answer"])
                if args.mode == "ans":
                    dataset["ground_truth"].append(result["ground_truth"])
                    dataset["citations"].append(result["citations"])
                    dataset["type"].append(result["type"])

                processed_count += 1

    execution_time = time.time() - start_time
    logger.info(f"TIME TAKEN >>>>>>>>>>> {execution_time} <<<<<<<<<<<")
    
    if args.mode == "ans":
        dataset['time'] = execution_time
    
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
