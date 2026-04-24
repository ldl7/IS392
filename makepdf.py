from fpdf import FPDF
import re

with open('E:/IS392/Module_13_Prompt_Engineering_Assignment.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace unicode chars
content = content.replace(chr(8212), '-')  # em-dash
content = content.replace(chr(8211), '-')   # en-dash  
content = content.replace(chr(8220), '"') # left double quote
content = content.replace(chr(8221), '"') # right double quote
content = content.replace(chr(8216), "'") # left single quote
content = content.replace(chr(8217), "'") # right single quote

# Clean formatting - remove table chars
clean = re.sub(r' \| ', '    ', content)
clean = re.sub(r'\|', '', clean)
clean = re.sub(r'^[\s\-]+$', '', clean, flags=re.MULTILINE)
clean = re.sub(r'^---+$', '', clean, flags=re.MULTILINE)
clean = clean.replace('```', '')
clean = re.sub(r'\n{3,}', '\n\n', clean)

# Create PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font('Arial', '', 10)

pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Module 13 Assignment - Prompt Engineering', ln=True, align='C')
pdf.ln(5)
pdf.set_font('Arial', '', 10)

for line in clean.split('\n'):
    line = line.strip()
    if not line:
        pdf.ln(2)
        continue
    if line.isupper() and len(line) < 50:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, line, ln=True)
        pdf.set_font('Arial', '', 10)
    elif line.startswith(('Part ', 'Task ', 'Prompt ')):
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, line, ln=True)
        pdf.set_font('Arial', '', 10)
    else:
        pdf.multi_cell(0, 5, line)

pdf.output('E:/IS392/Module_13_Prompt_Engineering_Assignment.pdf')
print('PDF created')
