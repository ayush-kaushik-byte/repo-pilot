from dataclasses import dataclass
from difflib import SequenceMatcher

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from repopilot.candidate_generator import CandidateFile


class SelectedFile(BaseModel):
    path: str = Field(
        description="Exact path of a selected repository file. "
                    "Must match one of the candidate paths exactly."
    )
    reason: str = Field(
        description="Why this file is relevant to the user's request."
    )


class FileSelection(BaseModel):
    selected_files: list[SelectedFile] = Field(
        description="Small set of candidate files relevant to the user's request."
    )


@dataclass
class FileSelector:

    TIER2_MIN_SCORE = 0.85
    TIER2_MIN_MARGIN = 0.10

    model_name: str = "qwen3.5:2b"
    temperature: float = 0.0

    def __post_init__(self):
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
        )

        self.structured_llm = self.llm.with_structured_output(
            FileSelection
        )

    async def select(
        self,
        query: str,
        candidates: list[CandidateFile],
    ) -> FileSelection:

        candidate_text = self._format_candidates(candidates)

        prompt = f"""
You are a repository analysis agent.

Your task is to identify which candidate files are most relevant
to answering the user's request.

You are NOT given the contents of the files.

Use ONLY:
1. The user's request.
2. The candidate file paths.
3. The deterministic signals provided for each candidate.

IMPORTANT RULES:

- Select only files from the candidate list.
- The path must match the candidate path EXACTLY.
- Never invent, modify, shorten, or normalize a file path.
- Prefer a small set of highly relevant files.
- Do not select files merely because they are entry points or configuration files.
- Think about which files are most likely to contain the information needed
  to answer the user's specific question.

User request:
{query}

Candidate files:
{candidate_text}

Select the files that should be retrieved next for deeper analysis.
"""

        return await self.structured_llm.ainvoke(prompt)

    @staticmethod
    def _format_candidates(
        candidates: list[CandidateFile],
    ) -> str:
        lines = []

        for index, candidate in enumerate(candidates, start=1):
            reasons = "; ".join(candidate.reasons)

            lines.append(
                f"{index}. PATH: {candidate.path}\n"
                f"   deterministic_score: {candidate.score:.1f}\n"
                f"   signals: {reasons}"
            )

        return "\n".join(lines)
    
    def validate_selection(
            self,
            selection: FileSelection,
            candidates: list[CandidateFile],
        ) -> FileSelection:

        # Build indexes once for this candidate set.
        # Exact path lookup: O(1) average.
        candidate_paths = {
            candidate.path
            for candidate in candidates
        }

        # Filename -> candidate paths.
        # This allows recovery when the LLM gets the directory wrong.
        filename_index: dict[str, list[str]] = {}

        for candidate in candidates:
            filename_index.setdefault(
                candidate.path.rsplit("/", 1)[-1],
                [],
            ).append(candidate.path)

        resolved_files = []

        for selected in selection.selected_files:

            # Case 1: exact match.
            if selected.path in candidate_paths:
                resolved_files.append(selected)
                continue

            print(
                f"WARNING: LLM selected a non-candidate path: "
                f"{selected.path}"
            )

            # Case 2: attempt deterministic recovery using filename.
            filename = selected.path.rsplit("/", 1)[-1]
            matches = filename_index.get(filename, [])

            if len(matches) == 1:
                resolved_path = matches[0]

                print(
                    f"RECOVERED: {selected.path} "
                    f"-> {resolved_path}"
                )

                resolved_files.append(
                    SelectedFile(
                        path=resolved_path,
                        reason=(
                            f"{selected.reason} "
                            f"(path recovered from unique filename match)"
                        ),
                    )
                )

            elif len(matches) > 1:
                print(
                    f"AMBIGUOUS: {selected.path} "
                    f"matches multiple candidate files: {matches}"
                )

            else:
                resolved_path = self._tier2_resolve(
                    selected.path,
                    candidates,
                )

                if resolved_path is not None:
                    print(
                        f"TIER2 RECOVERED: "
                        f"{selected.path} -> {resolved_path}"
                    )

                    resolved_files.append(
                        SelectedFile(
                            path=resolved_path,
                            reason=(
                                f"{selected.reason} "
                                f"(path recovered using "
                                f"Tier 2 similarity)"
                            ),
                        )
                    )
                else:
                    print(
                        f"UNRESOLVED: no confident candidate match "
                        f"found for {selected.path}"
                    )

        return FileSelection(
            selected_files=resolved_files
        )
    
    @staticmethod
    def _similarity(left: str, right: str) -> float:
        return SequenceMatcher(
            None,
            left,
            right,
        ).ratio()
    
    @staticmethod
    def _filename(path: str) -> str:
        return path.rsplit("/", 1)[-1]

    @staticmethod
    def _directory(path: str) -> str:
        return path.rsplit("/", 1)[0] if "/" in path else ""
    

    def _tier2_resolve(
        self,
        selected_path: str,
        candidates: list[CandidateFile],
    ) -> str | None:

        selected_filename = self._filename(selected_path)
        selected_directory = self._directory(selected_path)

        scored_candidates = []

        for candidate in candidates:
            candidate_filename = self._filename(candidate.path)
            candidate_directory = self._directory(candidate.path)

            filename_score = self._similarity(
                selected_filename,
                candidate_filename,
            )

            directory_score = self._similarity(
                selected_directory,
                candidate_directory,
            )

            total_score = (
                0.7 * filename_score
                + 0.3 * directory_score
            )

            scored_candidates.append(
                (total_score, candidate.path)
            )

        scored_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        if not scored_candidates:
            return None

        best_score, best_path = scored_candidates[0]

        second_score = (
            scored_candidates[1][0]
            if len(scored_candidates) > 1
            else 0.0
        )

        margin = best_score - second_score

        print(
            f"TIER2: {selected_path} -> "
            f"{best_path} "
            f"(score={best_score:.2f}, "
            f"margin={margin:.2f})"
        )

        if best_score < self.TIER2_MIN_SCORE:
            print(
                f"TIER2 REJECTED: score {best_score:.2f} "
                f"is below threshold "
                f"{self.TIER2_MIN_SCORE:.2f}"
            )
            return None

        if margin < self.TIER2_MIN_MARGIN:
            print(
                f"TIER2 AMBIGUOUS: margin {margin:.2f} "
                f"is below threshold "
                f"{self.TIER2_MIN_MARGIN:.2f}"
            )
            return None

        return best_path