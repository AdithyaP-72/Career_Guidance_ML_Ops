"""Resume → skills.   OWNED BY: <teammate> (see the brief in docs/PHASE2.md, "Resume → skills").

The API endpoint POST /extract-skills (src/serving/app.py) calls `resume_to_skills` below. Until it is
implemented, the endpoint answers 501 and the frontend shows "resume upload: coming soon".

The only fixed part is what the API hands in and expects back; everything inside is yours to design:

    resume_to_skills(data, filename, vocab, aliases) -> list[str]
        data      the uploaded file's bytes (PDF, DOCX or TXT; the API rejects anything over 2 MB)
        filename  original file name, e.g. "cv.pdf" (use the extension to pick a parser)
        vocab     set of skill names the role model knows, in OUR spelling (e.g. "react.js")
        aliases   {variant: canonical} from configs/skill_aliases.yaml, e.g. {"reactjs": "react.js"}
        returns   skills found, most relevant first. The API keeps only those in `vocab` and drops
                  duplicates, so returning extra is harmless, but returning nothing useful isn't.

Rules: process in memory only; never write the file to disk or log its text (personal data).
If the file can't be read (e.g. a scanned PDF with no text layer), raise ResumeError("<message for the user>").
"""

from __future__ import annotations


class ResumeError(ValueError):
    """The file was received but no text/skills could be read from it. The message is shown to the user."""


def resume_to_skills(data: bytes, filename: str, vocab: set[str], aliases: dict[str, str]) -> list[str]:
    raise NotImplementedError("resume upload is not built yet")
