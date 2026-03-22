from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from pypdf import PdfReader

CHAPTER_URL = "https://web.stanford.edu/~jurafsky/slp3/3.pdf"
CHAPTER_TITLE = "Chapter 3 – N-gram Language Models"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATOR_MODEL = "extractive-local"
OPTIONAL_API_MODEL = "gpt-4o-mini"
LOCAL_PDF_CANDIDATES = [
    Path("data/chapter3_ngram_language_models.pdf"),
    Path(__file__).resolve().parents[1] / "data" / "chapter3_ngram_language_models.pdf",
]


@dataclass
class Chunk:
    chunk_id: str
    page: int
    section: str
    text: str
    contextual_text: str


def resolve_pdf_path(explicit_path: str | None = None) -> str:
    if explicit_path and Path(explicit_path).exists():
        return str(Path(explicit_path))
    for candidate in LOCAL_PDF_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return download_chapter_pdf()


def download_chapter_pdf(
    url: str = CHAPTER_URL,
    save_path: str = "data/chapter3_ngram_language_models.pdf",
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        return save_path
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 A6-RAG-Assignment"},
    )
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z<])", text)
    return [p.strip() for p in parts if p.strip()]


def _is_valid_heading(title: str) -> bool:
    title = clean_text(title)
    words = title.split()
    if not title or len(words) > 14:
        return False
    lowered = title.lower()
    banned_starts = (
        "add an option",
        "you are given",
        "write out",
        "give three",
        "state",
        "compute",
        "show",
    )
    return not lowered.startswith(banned_starts)


def extract_pages_and_sections(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    pages: List[Dict[str, Any]] = []
    current_section = "3.0 Introduction"
    heading_pattern = re.compile(r"^(3(?:\.\d+)+)\s+(.+?)\s*$")

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in raw_lines:
            candidate = clean_text(line)
            match = heading_pattern.match(candidate)
            if match:
                number = match.group(1).strip()
                title = match.group(2).strip(" .")
                if _is_valid_heading(title):
                    current_section = f"{number} {title}"
        pages.append(
            {
                "page": page_idx,
                "section": current_section,
                "raw_text": raw_text,
                "clean_text": clean_text(raw_text),
            }
        )
    return pages


def window_words(words: List[str], size: int = 180, overlap: int = 40) -> List[str]:
    windows: List[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if len(window) < 60:
            continue
        windows.append(" ".join(window))
        if start + size >= len(words):
            break
    return windows


def summarize_for_prefix(chunk_text: str, max_words: int = 28) -> str:
    sentences = split_into_sentences(chunk_text)
    summary = sentences[0] if sentences else chunk_text
    summary_words = summary.split()
    if len(summary_words) > max_words:
        summary = " ".join(summary_words[:max_words]).rstrip(",;:") + "..."
    return summary


def build_context_prefix(section: str, page: int, summary: str) -> str:
    return f"This chunk from {CHAPTER_TITLE}, {section}, page {page}, discusses {summary}"


def build_chunks(
    pages: List[Dict[str, Any]],
    mode: str = "naive",
    chunk_size: int = 180,
    overlap: int = 40,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_data in pages:
        words = page_data["clean_text"].split()
        page_windows = window_words(words, size=chunk_size, overlap=overlap)
        for local_idx, window_text in enumerate(page_windows, start=1):
            summary = summarize_for_prefix(window_text)
            contextual_text = (
                build_context_prefix(page_data["section"], page_data["page"], summary)
                + "\n\n"
                + window_text
            )
            chunks.append(
                Chunk(
                    chunk_id=f"p{page_data['page']}_c{local_idx}",
                    page=page_data["page"],
                    section=page_data["section"],
                    text=window_text,
                    contextual_text=contextual_text if mode == "contextual" else window_text,
                )
            )
    return chunks


class VectorRetriever:
    def __init__(self, texts: List[str], model_name: str = DEFAULT_EMBED_MODEL):
        self.texts = texts
        self.model_name = model_name
        self.mode = None
        self.encoder = None
        self.matrix = None
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name)
            self.matrix = self.encoder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.mode = "sentence-transformer"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.encoder = TfidfVectorizer(stop_words="english")
            self.matrix = self.encoder.fit_transform(texts)
            self.mode = "tfidf"

    def search(self, query: str, top_k: int = 4) -> List[Tuple[int, float]]:
        if self.mode == "sentence-transformer":
            query_vec = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = np.dot(self.matrix, query_vec)
        else:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = self.encoder.transform([query])
            scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in ranked]


def lexical_sentence_score(sentence: str, query_terms: List[str]) -> float:
    sent_terms = re.findall(r"[A-Za-z0-9]+", sentence.lower())
    if not sent_terms:
        return 0.0
    overlap = sum(1 for t in sent_terms if t in query_terms)
    return overlap / math.sqrt(len(sent_terms))


def extractive_answer(query: str, retrieved_chunks: List[Chunk], max_sentences: int = 3) -> str:
    query_terms = re.findall(r"[A-Za-z0-9]+", query.lower())
    candidates = []
    for chunk in retrieved_chunks:
        for sentence in split_into_sentences(chunk.text):
            score = lexical_sentence_score(sentence, query_terms)
            if score > 0:
                candidates.append((score, sentence))
    if not candidates:
        fallback_sentences = split_into_sentences(retrieved_chunks[0].text)
        return fallback_sentences[0] if fallback_sentences else retrieved_chunks[0].text[:300]
    ranked = sorted(candidates, key=lambda x: x[0], reverse=True)
    selected, seen = [], set()
    for _, sentence in ranked:
        normalized = sentence.lower()
        if normalized not in seen:
            selected.append(sentence)
            seen.add(normalized)
        if len(selected) >= max_sentences:
            break
    return " ".join(selected)


def openai_answer(query: str, retrieved_chunks: List[Chunk], model: str = OPTIONAL_API_MODEL) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return extractive_answer(query, retrieved_chunks)
    try:
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        context_block = "\n\n".join(
            [
                f"[Source {i+1} | page {chunk.page} | {chunk.section}]\n{chunk.text}"
                for i, chunk in enumerate(retrieved_chunks)
            ]
        )
        prompt = f"""
You are answering questions strictly from one textbook chapter.
Use only the supplied sources. If the answer is not supported by the sources, say so briefly.

Question:
{query}

Sources:
{context_block}

Write a concise, accurate answer in 2-4 sentences.
"""
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return extractive_answer(query, retrieved_chunks)


def build_retriever_for_mode(mode: str = "naive", pdf_path: str | None = None) -> Tuple[List[Chunk], VectorRetriever]:
    resolved_pdf = resolve_pdf_path(pdf_path)
    pages = extract_pages_and_sections(resolved_pdf)
    chunks = build_chunks(pages, mode=mode)
    texts = [chunk.contextual_text for chunk in chunks]
    retriever = VectorRetriever(texts)
    return chunks, retriever


def answer_question(
    query: str,
    chunks: List[Chunk],
    retriever: VectorRetriever,
    top_k: int = 4,
    generator: str = "extractive",
) -> Dict[str, Any]:
    hits = retriever.search(query, top_k=top_k)
    retrieved = [chunks[idx] for idx, _ in hits]
    answer = extractive_answer(query, retrieved) if generator == "extractive" else openai_answer(query, retrieved)
    sources = [
        {
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "section": chunk.section,
            "score": round(score, 4),
            "text": chunk.text[:1200],
        }
        for (idx, score), chunk in zip(hits, retrieved)
    ]
    return {"answer": answer, "sources": sources}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-−][A-Za-z0-9]+)?", text.lower())


def make_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(pred: str, ref: str, n: int = 1) -> float:
    from collections import Counter

    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    pred_ngrams = Counter(make_ngrams(pred_tokens, n))
    ref_ngrams = Counter(make_ngrams(ref_tokens, n))
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum((pred_ngrams & ref_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_rows(rows: List[Dict[str, str]], answer_key: str) -> Dict[str, float]:
    r1 = np.mean([rouge_n_f1(row[answer_key], row["ground_truth_answer"], 1) for row in rows])
    r2 = np.mean([rouge_n_f1(row[answer_key], row["ground_truth_answer"], 2) for row in rows])
    rl = np.mean([rouge_l_f1(row[answer_key], row["ground_truth_answer"]) for row in rows])
    return {"ROUGE-1": float(r1), "ROUGE-2": float(r2), "ROUGE-L": float(rl)}
