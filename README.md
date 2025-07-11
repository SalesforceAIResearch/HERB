# Benchmarking Deep Search over Heterogeneous Enterprise Data

**Paper:** [Arxiv Link](https://arxiv.org/abs/2506.23139)  
**Dataset:** [Huggingface Link](https://huggingface.co/datasets/Salesforce/HERB)

We present a new benchmark for evaluating Deep Search--a realistic and complex form of retrieval-augmented generation (RAG) that requires source-aware, multi-hop reasoning over diverse, sparsed, but related sources. These include documents, meeting transcripts, Slack messages, GitHub, and URLs, which vary in structure and often contain human-to-human interactions. We build it using a synthetic data pipeline that simulates business workflows across product planning, development, and support stages, generating interconnected content with realistic noise and multi-hop questions with guaranteed ground-truth answers. We release our benchmark with both answerable and unanswerable queries, and retrieval pool of 39,190 enterprise artifacts, enabling fine-grained evaluation of long-context LLM and RAG systems. Our experiments reveal that even the best-performing agentic RAG methods achieve an average performance score of 32.96 on our benchmark. With further analysis, we highlight retrieval as the main bottleneck: existing methods struggle to conduct deep searches and retrieve all necessary evidence. Consequently, they often reason over partial context, leading to significant performance degradation.

## Usage

You need to have an OpenAI key `export OPENAI_API_KEY='yourkey'` or Together AI key `export TOGETHER_API_KEY='yourkey'` or set up [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart?usertype=apikey)

```bash
cd code
```

### 1. RAG over all data

```bash
python rag.py --mode ans    # Answerable Evaluation
python rag.py --mode unans  # Unanswerable Evaluation
```

### 2. Product-Specific RAG

```bash
python product_rag.py --mode ans    # Answerable Evaluation
python product_rag.py --mode unans  # Unanswerable Evaluation
```

### 3. Long Context Setting

```bash
python long_context_eval.py --mode ans    # Answerable Evaluation
python long_context_eval.py --mode unans  # Unanswerable Evaluation
```

### 4. Oracle Setting

```bash
python oracle_eval.py   # Answerable Evaluation
```

### 5. Evaluation

```bash
python evaluate.py --output_file {output_file_name}
```

## Ethical Considerations
HERB was generated using GPT-4o and should not be used to develop models that compete with OpenAI.

This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people's lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.

## Citation

```bibtex
@article{choubey2025benchmarkingdeepsearchheterogeneous,
    title={Benchmarking Deep Search over Heterogeneous Enterprise Data}, 
    author={Prafulla Kumar Choubey and Xiangyu Peng and Shilpa Bhagavath and Kung-Hsiang Huang and Caiming Xiong and Chien-Sheng Wu},
    year={2025},
    eprint={2506.23139},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2506.23139}
}
```