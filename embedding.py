"""
evaluate_retrieval.py — Answer-based retrieval evaluation.

The evaluation set contains only:

[
  {
    "question": "...",
    "answer": "..."
  }
]

For every question, the evaluator:

1. Retrieves the top-k chunks from Chroma.
2. Displays the retrieved chunks.
3. Compares each retrieved chunk against the reference answer
   using embedding similarity.
4. Reports the best similarity score and average similarity.

This evaluates whether the retrieved context contains information
semantically related to the expected answer.

Usage:
    python evaluate_retrieval.py <vectorstore_dir> <eval_set.json>
"""

import sys
import json
import numpy as np

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
            np.linalg.norm(a) *
            np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# --------------------------------------------------------------------------
# Answer-based retrieval evaluation
# --------------------------------------------------------------------------

def evaluate_question(
        question,
        reference_answer,
        vectorstore,
        embeddings,
        k=5,
):
    """
    Retrieve top-k chunks and compare them with the reference answer.
    """

    results = vectorstore.similarity_search(
        question,
        k=k,
    )

    if not results:
        return {
            "results": [],
            "similarities": [],
            "best_similarity": 0.0,
            "average_similarity": 0.0,
        }

    # Embed the expected answer
    answer_vector = embeddings.embed_query(
        reference_answer
    )

    similarities = []

    for document in results:

        chunk_vector = embeddings.embed_query(
            document.page_content
        )

        similarity = cosine_similarity(
            answer_vector,
            chunk_vector,
        )

        similarities.append(similarity)

    best_similarity = max(
        similarities
    )

    average_similarity = (
            sum(similarities)
            / len(similarities)
    )

    return {
        "results": results,
        "similarities": similarities,
        "best_similarity": best_similarity,
        "average_similarity": average_similarity,
    }


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------

def evaluate(
        vectorstore_dir,
        eval_set_path,
        k=5,
        collection_name="docchat",
):
    """
    Evaluate retrieval using question + answer pairs.
    """

    # ----------------------------------------------------------------------
    # Load evaluation set
    # ----------------------------------------------------------------------

    with open(
            eval_set_path,
            "r",
            encoding="utf-8",
    ) as f:
        eval_set = json.load(f)

    # ----------------------------------------------------------------------
    # Load embedding model
    #
    # IMPORTANT:
    # This must match the model used during ingestion.
    # ----------------------------------------------------------------------

    embeddings = FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # ----------------------------------------------------------------------
    # Open Chroma
    # ----------------------------------------------------------------------

    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    # ----------------------------------------------------------------------
    # Check database 
    # ----------------------------------------------------------------------

    try:
        count = vectorstore._collection.count()
        print(
            f"\nChroma collection: {collection_name}"
        )
        print(
            f"Documents in collection: {count}"
        )
    except Exception:
        pass

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANSWER-BASED RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )

    best_scores = []
    average_scores = []

    # ----------------------------------------------------------------------
    # Evaluate each question
    # ----------------------------------------------------------------------

    for index, item in enumerate(
            eval_set,
            start=1,
    ):

        question = item["question"]
        reference_answer = item["answer"]

        evaluation = evaluate_question(
            question=question,
            reference_answer=reference_answer,
            vectorstore=vectorstore,
            embeddings=embeddings,
            k=k,
        )

        results = evaluation["results"]
        similarities = evaluation["similarities"]

        best_similarity = (
            evaluation["best_similarity"]
        )

        average_similarity = (
            evaluation["average_similarity"]
        )

        best_scores.append(
            best_similarity
        )

        average_scores.append(
            average_similarity
        )

        # ------------------------------------------------------------------
        # Status
        #
        # These are rough diagnostic thresholds, not absolute measures.
        # ------------------------------------------------------------------

        if best_similarity >= 0.70:
            status = "✅"

        elif best_similarity >= 0.55:
            status = "🟡"

        else:
            status = "❌"

        print(
            f"\n{status} Question {index}"
        )

        print(
            f"\nQ: {question}"
        )

        print(
            f"\nReference answer:"
        )

        print(
            f"  {reference_answer}"
        )

        print(
            "\nRetrieved chunks:"
        )

        if not results:

            print(
                "  No results retrieved."
            )

        else:

            for rank, (
                    document,
                    similarity,
            ) in enumerate(
                zip(
                    results,
                    similarities,
                ),
                start=1,
            ):

                chapter = document.metadata.get(
                    "chapter",
                    "Unknown",
                )

                print(
                    f"\n  [{rank}] "
                    f"similarity={similarity:.3f}"
                )

                print(
                    f"      chapter: {chapter}"
                )

                print(
                    f"      page: "
                    f"{document.metadata.get('start_page', 'Unknown')}"
                )

                print(
                    f"      text: "
                    f"{document.page_content[:500]}"
                )

        print(
            f"\n  Best similarity: "
            f"{best_similarity:.3f}"
        )

        print(
            f"  Average similarity: "
            f"{average_similarity:.3f}"
        )

        print(
            "-" * 70
        )

    # ----------------------------------------------------------------------
    # Overall statistics
    # ----------------------------------------------------------------------

    if best_scores:

        mean_best = (
                sum(best_scores)
                / len(best_scores)
        )

        mean_average = (
                sum(average_scores)
                / len(average_scores)
        )

        # Percentage of questions where at least
        # one retrieved chunk has reasonably strong
        # semantic similarity to the reference answer.

        strong_retrieval = sum(
            1
            for score in best_scores
            if score >= 0.70
        )

        retrieval_rate = (
                strong_retrieval
                / len(best_scores)
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "OVERALL RESULTS"
        )

        print(
            "=" * 70
        )

        print(
            f"Questions evaluated: "
            f"{len(best_scores)}"
        )

        print(
            f"Mean best similarity: "
            f"{mean_best:.3f}"
        )

        print(
            f"Mean average similarity: "
            f"{mean_average:.3f}"
        )

        print(
            f"Strong retrieval rate: "
            f"{retrieval_rate:.3f}"
        )

        print(
            f"Strong retrieval: "
            f"{strong_retrieval}/"
            f"{len(best_scores)}"
        )

        print(
            "=" * 70
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(__doc__)
        sys.exit(1)

    evaluate(
        vectorstore_dir=sys.argv[1],
        eval_set_path=sys.argv[2],
    )