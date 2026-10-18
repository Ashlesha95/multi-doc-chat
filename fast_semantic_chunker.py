"""
fast_semantic_chunker.py

A faster semantic chunker inspired by LangChain's SemanticChunker.

Main difference:
    LangChain SemanticChunker
        text
         ↓
        sentences
         ↓
        embedding calls
         ↓
        cosine distances
         ↓
        percentile breakpoint
         ↓
        chunks

    FastSemanticChunker
        text
         ↓
        sentences
         ↓
        BATCH embedding with FastEmbed
         ↓
        cosine distances
         ↓
        percentile breakpoint
         ↓
        chunks

The semantic chunking strategy remains the same:
    - sentence based
    - cosine distance between adjacent sentences
    - percentile breakpoint detection
    - default breakpoint threshold = 90

FastEmbed is used directly so we can control:
    - batch_size
    - parallel embedding workers

This is intended for offline document ingestion.
"""

import re
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding

from langchain_core.documents import Document


class FastSemanticChunker:
    """
    Semantic chunker using FastEmbed directly.

    The algorithm is intentionally close to LangChain's SemanticChunker,
    but embedding is performed in batches using FastEmbed.

    Parameters
    ----------
    model_name:
        FastEmbed model name.

    breakpoint_threshold_amount:
        Percentile used to detect semantic breakpoints.

        Example:
            90 means only the largest 10% of semantic jumps
            become chunk boundaries.

    batch_size:
        Number of sentences embedded together.

    parallel:
        FastEmbed data-parallel workers.

        None:
            Use ONNX Runtime's normal threading.

        0:
            Use all available CPU cores.

        >1:
            Use FastEmbed data parallelism.

    sentence_split_regex:
        Regex used to split text into sentences.

    min_sentences:
        Minimum number of sentences needed before semantic
        chunking is meaningful.
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-small-en-v1.5",
            breakpoint_threshold_amount: float = 90.0,
            batch_size: int = 32,
            parallel: int | None = None,
            sentence_split_regex: str = r"(?<=[.?!])\s+",
            min_sentences: int = 2,
    ):
        if not 0 <= breakpoint_threshold_amount <= 100:
            raise ValueError(
                "breakpoint_threshold_amount must be between 0 and 100."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        self.model_name = model_name
        self.breakpoint_threshold_amount = (
            breakpoint_threshold_amount
        )
        self.batch_size = batch_size
        self.parallel = parallel
        self.sentence_split_regex = sentence_split_regex
        self.min_sentences = min_sentences

        print(
            f"Loading FastEmbed model: {model_name}"
        )

        self.model = TextEmbedding(
            model_name=model_name
        )

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def _split_sentences(
            self,
            text: str,
    ) -> list[str]:
        """
        Split text into sentences.

        This follows the same default sentence-boundary idea used by
        LangChain SemanticChunker.
        """

        text = text.strip()

        if not text:
            return []

        sentences = re.split(
            self.sentence_split_regex,
            text,
        )

        # Clean whitespace and discard empty pieces.
        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        return sentences

    # ------------------------------------------------------------------
    # Batched embedding
    # ------------------------------------------------------------------

    def _embed_sentences(
            self,
            sentences: list[str],
    ) -> np.ndarray:
        """
        Embed all sentences using FastEmbed.

        FastEmbed handles batching internally.

        batch_size controls how many sentences are processed
        together.

        parallel controls FastEmbed's data-parallel execution.
        """

        if not sentences:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        embeddings = list(
            self.model.embed(
                sentences,
                batch_size=self.batch_size,
                parallel=self.parallel,
            )
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Cosine distances
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_distances(
            embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate cosine distance between adjacent sentence embeddings.

        distance = 1 - cosine_similarity
        """

        if len(embeddings) < 2:
            return np.array(
                [],
                dtype=np.float32,
            )

        # Normalize vectors.
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        # Avoid division by zero.
        norms = np.maximum(
            norms,
            1e-12,
        )

        normalized = (
                embeddings / norms
        )

        # Adjacent cosine similarities.
        similarities = np.sum(
            normalized[:-1] * normalized[1:],
            axis=1,
            )

        distances = 1.0 - similarities

        return distances

    # ------------------------------------------------------------------
    # Percentile breakpoint
    # ------------------------------------------------------------------

    def _find_breakpoints(
            self,
            distances: np.ndarray,
    ) -> list[int]:
        """
        Find sentence indices where semantic distance is above
        the configured percentile threshold.

        Example:

            sentences:
                0  1  2  3  4  5

            distances:
                └─┴─┴─┴─┴─┘

            breakpoint at distance between 2 and 3

            → split after sentence 2
        """

        if len(distances) == 0:
            return []

        threshold = np.percentile(
            distances,
            self.breakpoint_threshold_amount,
        )

        breakpoints = [
            index
            for index, distance in enumerate(distances)
            if distance > threshold
        ]

        return breakpoints

    # ------------------------------------------------------------------
    # Build chunks
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chunks(
            sentences: list[str],
            breakpoints: list[int],
    ) -> list[str]:
        """
        Convert breakpoint indices into text chunks.

        A breakpoint at index i means:
            sentence i
            and sentence i+1
        are separated.
        """

        if not sentences:
            return []

        if not breakpoints:
            return [
                " ".join(sentences).strip()
            ]

        chunks = []

        start = 0

        for breakpoint in breakpoints:

            end = breakpoint + 1

            chunk = " ".join(
                sentences[start:end]
            ).strip()

            if chunk:
                chunks.append(chunk)

            start = end

        # Final chunk.
        final_chunk = " ".join(
            sentences[start:]
        ).strip()

        if final_chunk:
            chunks.append(final_chunk)

        return chunks

    # ------------------------------------------------------------------
    # Main split
    # ------------------------------------------------------------------

    def split_text(
            self,
            text: str,
    ) -> list[str]:
        """
        Split text into semantic chunks.
        """

        sentences = self._split_sentences(
            text
        )

        if not sentences:
            return []

        # If there are too few sentences, semantic comparison
        # doesn't provide useful information.
        if len(sentences) < self.min_sentences:
            return [
                " ".join(sentences).strip()
            ]

        # --------------------------------------------------------------
        # Batch embedding happens here.
        # --------------------------------------------------------------

        embeddings = self._embed_sentences(
            sentences
        )

        # --------------------------------------------------------------
        # Calculate semantic distances.
        # --------------------------------------------------------------

        distances = self._cosine_distances(
            embeddings
        )

        # --------------------------------------------------------------
        # Find semantic breakpoints.
        # --------------------------------------------------------------

        breakpoints = self._find_breakpoints(
            distances
        )

        # --------------------------------------------------------------
        # Build final chunks.
        # --------------------------------------------------------------

        return self._build_chunks(
            sentences,
            breakpoints,
        )

    # ------------------------------------------------------------------
    # Document API
    # ------------------------------------------------------------------

    def create_documents(
            self,
            texts: list[str],
            metadatas: list[dict] | None = None,
    ) -> list[Document]:
        """
        Convert input texts into LangChain Documents.
        """

        documents = []

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(metadatas) != len(texts):
            raise ValueError(
                "metadatas must have the same length as texts."
            )

        for text, metadata in zip(
                texts,
                metadatas,
        ):

            chunks = self.split_text(
                text
            )

            for chunk in chunks:

                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=dict(metadata),
                    )
                )

        return documents

    def split_documents(
            self,
            documents: Iterable[Document],
    ) -> list[Document]:
        """
        Split LangChain Documents while preserving metadata.
        """

        output = []

        for document in documents:

            chunks = self.split_text(
                document.page_content
            )

            for chunk in chunks:

                output.append(
                    Document(
                        page_content=chunk,
                        metadata=dict(
                            document.metadata
                        ),
                    )
                )

        return output

