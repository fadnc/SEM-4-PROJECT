"""
Extract text from Smart ICU Assistant PDF
"""
import PyPDF2

def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
    return text

if __name__ == "__main__":
    pdf_path = "Smart ICU Assistant.pdf"
    text = extract_pdf_text(pdf_path)
    
    # Save to text file
    with open("project_requirements.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Extracted {len(text)} characters from PDF")
    print("\n" + "="*60)
    print(text)
