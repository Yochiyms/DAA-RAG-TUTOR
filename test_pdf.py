from pypdf import PdfReader

reader = PdfReader("daa_book.pdf")
print(f"Number of pages: {len(reader.pages)}")
print(f"Is encrypted: {reader.is_encrypted}")

if len(reader.pages) > 0:
    text = reader.pages[0].extract_text()
    print(f"First page text preview: {text[:300]}")
else:
    print("No pages found by pypdf either")