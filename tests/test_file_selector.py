from repopilot.candidate_generator import CandidateFile
from repopilot.file_selector import (
    FileSelection,
    FileSelector,
    SelectedFile,
)


def make_candidate(path: str) -> CandidateFile:
    return CandidateFile(
        path=path,
        score=1.0,
        reasons=["test candidate"],
    )


def make_selection(path: str) -> FileSelection:
    return FileSelection(
        selected_files=[
            SelectedFile(
                path=path,
                reason="test selection",
            )
        ]
    )


def test_exact_path_is_accepted():
    candidates = [
        make_candidate("mysite/blog/views.py"),
        make_candidate("mysite/blog/models.py"),
    ]

    selection = make_selection("mysite/blog/views.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 1
    assert result.selected_files[0].path == "mysite/blog/views.py"


def test_unique_filename_is_recovered():
    candidates = [
        make_candidate("mysite/blog/views.py"),
        make_candidate("mysite/blog/models.py"),
    ]

    selection = make_selection("mysite/views.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 1
    assert result.selected_files[0].path == "mysite/blog/views.py"


def test_ambiguous_filename_is_rejected():
    candidates = [
        make_candidate("mysite/blog/views.py"),
        make_candidate("mysite/api/views.py"),
    ]

    selection = make_selection("mysite/views.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 0


def test_unresolved_path_is_rejected():
    candidates = [
        make_candidate("mysite/blog/views.py"),
        make_candidate("mysite/blog/models.py"),
    ]

    selection = make_selection("mysite/foobar.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 0

def test_tier2_recovers_high_similarity_filename():
    candidates = [
        make_candidate("mysite/blog/views.py"),
        make_candidate("mysite/blog/models.py"),
    ]

    selection = make_selection("mysite/blog/view.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 1
    assert result.selected_files[0].path == "mysite/blog/views.py"


def test_tier2_can_recover_strong_filename_match():
    candidates = [
        make_candidate("mysite/api/views.py"),
        make_candidate("mysite/models/user.py"),
    ]

    selection = make_selection("mysite/blog/view.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 1
    assert result.selected_files[0].path == "mysite/api/views.py"


def test_tier2_rejects_close_ambiguous_matches():
    candidates = [
        make_candidate("mysite/blog/views.py"),
        make_candidate("mysite/api/views.py"),
    ]

    selection = make_selection("mysite/view.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 0


def test_tier2_rejects_weak_similarity():
    candidates = [
        make_candidate("mysite/blog/views.py"),
        make_candidate("mysite/blog/models.py"),
    ]

    selection = make_selection("mysite/payment.py")

    selector = FileSelector.__new__(FileSelector)

    result = selector.validate_selection(
        selection,
        candidates,
    )

    assert len(result.selected_files) == 0