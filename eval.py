import time
import json
import re
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_chain import rewriter, understand_and_rerank, format_context
from config import settings


CHAT_ID = "test"
RUN_LABEL = "hybrid_eval_06_04_reranker_querybreakdown"
K = 4 

# Replace model with your preferred model for generation and judging
# Initialize Pipeline
generator_llm = ChatOllama(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.2)
judge_llm = ChatOllama(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.0)

"""
TEST QUESTIONS FORMAT (test_questions.json)

[
    {
        "question": "What is the scaling factor applied in the scaled dot-product attention?",
        "ground_truth": "The dot products are scaled by a factor of 1/sqrt(d_k) to prevent the gradients from becoming too small.",
        "expected_source": "Attention is all you need.pdf",
        "expected_page": "4"
    },
    {....},..... ]
"""


with open("test_questions.json", "r", encoding="utf-8") as f:
    EVAL_DATA = json.load(f)

print("=" * 80)
print(f" Starting Evaluation: {RUN_LABEL}")
print(f" Dataset Size : {len(EVAL_DATA)}")
print(f" Retriever    : Hybrid")
print(f" Generator    : qwen2.5:7b-instruct-q4_K_M")
print(f"  Judge        : qwen2.5:7b-instruct-q4_K_M")
print(f" Top-K         : {K}")
print("=" * 80)

gen_prompt =  ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a Retrieval-Augmented Generation (RAG) assistant.

        Answer the user's question ONLY using the retrieved context below.

        If the answer is present in the context, answer it directly.

        Ignore irrelevant chunks.

        If multiple chunks support the answer, combine them.

        If the answer is truly not contained in the context, reply exactly:

        "I could not find that information in the uploaded documents."

        ========================
        Retrieved Context
        ========================

        {context}
        """),
            ("human", "{question}")
        ])

eval_prompt = ChatPromptTemplate.from_template("""You are a strict JSON-only judge evaluating a RAG system.
    Question: {question}
    Ground Truth: {ground_truth}
    Prediction: {prediction}

    Respond ONLY with valid JSON containing integer values 0 or 1:
    {{
        "faithfulness": 1 if prediction does not hallucinate beyond context else 0,
        "relevance": 1 if prediction correctly answers the question matching ground truth else 0
    }}""")


def run_evaluation():
    results = []
    agg = {
        "hits": 0, "mrr_sum": 0.0, "faith_sum": 0, "rel_sum": 0,
        "time_retrieval": 0.0, "time_generation": 0.0, "time_judge": 0.0
    }
    
    for idx, item in enumerate(EVAL_DATA):
        q = item["question"]
        expected_src = item["expected_source"]
        expected_pg = str(item.get("expected_page", ""))
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(EVAL_DATA)}] {q}")        
        # 1. RETRIEVAL (Isolated Timing)
        t0 = time.time()

        rewritten_query = rewriter(
            settings.default_provider,
            settings.default_model
        ).invoke(
            {
                "history": "",
                "question": q,
            }
        )

        rewritten_query, reranked = understand_and_rerank(
            CHAT_ID,
            rewritten_query,
        )
        t_ret = time.time() - t0
        hit = 0
        mrr = 0.0
        retrieved_meta = []
        
        # Calculate Hit@K, Recall@K, MRR & Extract Metadata
        for rank, doc in enumerate(reranked):
            src = str(doc.metadata.get("source", ""))
            pg = str(doc.metadata.get("page", ""))
            retrieved_meta.append({"source": src, "page": pg, "rank": rank + 1})
            
            if hit == 0 and expected_src in src and expected_pg == pg:
                hit = 1  # For single-target RAG, Hit@K == Recall@K
                mrr = 1.0 / (rank + 1)
        
        context_str = format_context(reranked)
        
        #GENERATION
        t1 = time.time()
        prediction = (
        gen_prompt
        | generator_llm
        | StrOutputParser()
    ).invoke(
        {
            "context": context_str,
            "question": rewritten_query,
        }
    )
        preview = prediction.replace("\n", " ")[:150]   
        print(f"   💬 {preview}...")
        t_gen = time.time() - t1
        
        #JUDGE 
        t2 = time.time()
        eval_resp = (eval_prompt | judge_llm).invoke({
            "question": q, 
            "ground_truth": item["ground_truth"], 
            "prediction": prediction
        }).content
        t_jdg = time.time() - t2
        
        # Parse JSON from Judge safely
        try:
            match = re.search(r'\{.*\}', eval_resp, re.DOTALL)
            scores = json.loads(match.group(0)) if match else {"faithfulness": 0, "relevance": 0}
        except Exception:
            scores = {"faithfulness": 0, "relevance": 0}
            
        print(
    f"Hit={hit} | "
    f"MRR={mrr:.2f} | "
    f"Faith={scores['faithfulness']} | "
    f"Rel={scores['relevance']}"
)

        # Log individual result
        results.append({
            "question": q,
            "prediction": prediction,
            "expected_match": {"source": expected_src, "page": expected_pg},
            "retrieved_metadata": retrieved_meta,
            "metrics": {
                "hit_at_k": hit,
                "recall_at_k": hit, 
                "mrr": mrr,
                "faithfulness": scores.get("faithfulness", 0),
                "relevance": scores.get("relevance", 0),
            },
            "latency": {
                "retrieval_sec": round(t_ret, 3),
                "generation_sec": round(t_gen, 3),
                "judge_sec": round(t_jdg, 3)
            }
        })
        
        # Accumulate metrics
        agg["hits"] += hit
        agg["mrr_sum"] += mrr
        agg["faith_sum"] += scores.get("faithfulness", 0)
        agg["rel_sum"] += scores.get("relevance", 0)
        agg["time_retrieval"] += t_ret
        agg["time_generation"] += t_gen
        agg["time_judge"] += t_jdg


    n = len(EVAL_DATA) if len(EVAL_DATA) > 0 else 1
    final_report = {
        "run_label": RUN_LABEL,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top_k": K,
        "total_questions": len(EVAL_DATA),
        "aggregates": {
            "hit_rate_at_k": round(agg["hits"] / n, 3),
            "recall_at_k": round(agg["hits"] / n, 3),
            "mrr": round(agg["mrr_sum"] / n, 3),
            "faithfulness": round(agg["faith_sum"] / n, 3),
            "relevance": round(agg["rel_sum"] / n, 3)
        },
        "average_latency_sec": {
            "retrieval": round(agg["time_retrieval"] / n, 3),
            "generation": round(agg["time_generation"] / n, 3),
            "judge": round(agg["time_judge"] / n, 3)
        },
        "details": results
    }

    # Save to disk
    output_filename = f"eval_results_{RUN_LABEL}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\n Results saved to {output_filename}")

if __name__ == "__main__":
    run_evaluation()