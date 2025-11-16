import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re  # noqa: E402
from typing import List, Tuple  # noqa: E402

import tiktoken  # noqa: E402


class TextChunker:
    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 100, min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_text(
        self, text: str, document_name: str = "faq_document.txt"
    ) -> List[Tuple[str, dict]]:
        qa_chunks = self._chunk_by_qa_structure(text)
        if len(qa_chunks) >= 20:
            return qa_chunks
        return self._recursive_chunk(text)

    def _chunk_by_qa_structure(
        self, text: str, document_name: str = "faq_document.txt"
    ) -> List[Tuple[str, dict]]:
        chunks = []
        qa_pattern = "\\n\\n(?=Q:)"
        sections = re.split(qa_pattern, text)
        for i, section in enumerate(sections):
            if section.strip():
                lines = section.strip().split("\n")
                question = ""
                answer = ""
                for line in lines:
                    if line.startswith("Q:"):
                        question = line[2:].strip()
                    elif line.startswith("A:"):
                        answer = line[2:].strip()
                    elif answer:
                        answer += " " + line.strip()
                if question and answer:
                    chunk_text = f"Q: {question}\nA: {answer}"
                    text_lines = text.split("\n")
                    start_line = 1
                    for line_idx, line in enumerate(text_lines):
                        if question in line or (
                            len(question) > 20 and question[:20] in line
                        ):
                            start_line = line_idx + 1
                            break
                    key_phrases = self._extract_key_phrases(chunk_text)
                    metadata = {
                        "chunk_id": f"qa_{i}",
                        "chunk_type": "qa_pair",
                        "question": question,
                        "answer": answer,
                        "document_name": document_name,
                        "line_numbers": {
                            "start": start_line,
                            "end": start_line + len(section.split("\n")),
                        },
                        "key_phrases": key_phrases,
                        "section_type": "Q&A",
                        "token_count": len(self.tokenizer.encode(chunk_text)),
                    }
                    chunks.append((chunk_text, metadata))
        if chunks:
            topic_chunks = self._create_topic_chunks(text)
            chunks.extend(topic_chunks)
        return chunks

    def _create_topic_chunks(self, text: str) -> List[Tuple[str, dict]]:
        chunks = []
        heading_pattern = "\\n\\n([A-Z][A-Z\\s]+)\\n\\n"
        sections = re.split(heading_pattern, text)
        current_topic = "GENERAL"
        for i, section in enumerate(sections):
            if i % 2 == 1:
                current_topic = section.strip()
            elif section.strip():
                subsections = self._split_by_size(section, current_topic)
                chunks.extend(subsections)
        return chunks

    def _recursive_chunk(self, text: str) -> List[Tuple[str, dict]]:
        chunks = []
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

        def split_text(text: str, sep_index: int = 0) -> List[str]:
            if sep_index >= len(separators):
                return [text]
            separator = separators[sep_index]
            parts = text.split(separator)
            result = []
            current_chunk = ""
            for part in parts:
                test_chunk = current_chunk + separator + part if current_chunk else part
                if len(self.tokenizer.encode(test_chunk)) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    if len(self.tokenizer.encode(part)) > self.chunk_size:
                        result.extend(split_text(part, sep_index + 1))
                        current_chunk = ""
                    else:
                        current_chunk = part
            if current_chunk:
                result.append(current_chunk)
            return result

        text_chunks = split_text(text)
        for i, chunk in enumerate(text_chunks):
            if len(chunk.strip()) >= self.min_chunk_size:
                if i > 0 and self.chunk_overlap > 0:
                    prev_chunk = text_chunks[i - 1]
                    overlap_text = (
                        prev_chunk[-self.chunk_overlap :]
                        if len(prev_chunk) > self.chunk_overlap
                        else prev_chunk
                    )
                    chunk = overlap_text + " " + chunk
                metadata = {
                    "chunk_id": f"chunk_{i}",
                    "chunk_type": "recursive",
                    "token_count": len(self.tokenizer.encode(chunk)),
                    "position": i,
                }
                chunks.append((chunk.strip(), metadata))
        return chunks

    def _split_by_size(self, text: str, topic: str) -> List[Tuple[str, dict]]:
        chunks = []
        words = text.split()
        current_chunk = ""
        chunk_index = 0
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            if len(self.tokenizer.encode(test_chunk)) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk_id = f"topic_{topic.lower().replace(' ', '_')}_{chunk_index}"
                    metadata = {
                        "chunk_id": chunk_id,
                        "chunk_type": "topic_section",
                        "topic": topic,
                        "token_count": len(self.tokenizer.encode(current_chunk)),
                    }
                    chunks.append((current_chunk.strip(), metadata))
                    chunk_index += 1
                current_chunk = word
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            metadata = {
                "chunk_id": f"topic_{topic.lower().replace(' ', '_')}_{chunk_index}",
                "chunk_type": "topic_section",
                "topic": topic,
                "token_count": len(self.tokenizer.encode(current_chunk)),
            }
            chunks.append((current_chunk.strip(), metadata))
        return chunks

    def _extract_key_phrases(self, text: str) -> List[str]:
        import re

        key_phrases = []
        numbers_pattern = (
            "\\b\\d+(?:\\.\\d+)?\\s*(?:days?|hours?|weeks?|months?|years?|"
            "dollars?|\\$|%|percent)\\b"
        )
        numbers = re.findall(numbers_pattern, text, re.IGNORECASE)
        key_phrases.extend(numbers)
        policy_terms = [
            "\\b(?:vacation|PTO|sick leave|health insurance|401k|probation|"
            "overtime|remote work)\\b",
            "\\b(?:performance review|direct deposit|payroll|benefits|"
            "enrollment)\\b",
            "\\b(?:HR portal|manager approval|documentation required|"
            "eligibility)\\b",
        ]
        for pattern in policy_terms:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_phrases.extend(matches)
        quoted_text = re.findall('"([^"]*)"', text)
        key_phrases.extend(quoted_text)
        pattern = "[^.!?]*(?:must|required|mandatory|policy|procedure)" "[^.!?]*[.!?]"
        important_sentences = re.findall(pattern, text, re.IGNORECASE)
        key_phrases.extend([s.strip() for s in important_sentences])
        key_phrases = list(
            set([phrase.strip() for phrase in key_phrases if phrase.strip()])
        )
        return key_phrases[:10]
