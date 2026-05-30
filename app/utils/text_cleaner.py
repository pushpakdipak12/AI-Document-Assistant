import re

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and non-printable characters.
    """

    # Remove non-printable characters
    cleaned_text = re.sub(r'[^\x20-\x7E]+', ' ', text)

    # Replace multiple whitespace with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')


    return cleaned_text