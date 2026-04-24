
from fpdf import FPDF
import re

# Read the text file
with open('E:/IS392/Module_13_Prompt_Engineering_Assignment.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Clean up formatting characters
clean = content

# Convert tables to cleaner format - replace pipes with spacing
clean = re.sub(r' \| ', '    ', clean)  # Table cell separators -> spaces
clean = re.sub(r'\|', '', clean)  # Remove stray pipes

# Remove table separator lines (lines with only dashes and spaces)
clean = re.sub(r'^[\s\-]+$', '', clean, flags=re.MULTILINE)

# Remove horizontal rules
clean = re.sub(r'^---+$', '', clean, flags=re.MULTILINE)

# Remove backticks from code blocks but keep content
clean = clean.replace('\', '')

# Clean up multiple blank lines
clean = re.sub(r'
{3,}', '

', clean)

# Create PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Use a standard font
pdf.set_font('Arial', '', 11)

# Add title
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Module 13 Assignment - Prompt Engineering', ln=True, align='C')
pdf.ln(5)

# Reset font for body
pdf.set_font('Arial', '', 10)

# Process and add content
lines = clean.split('
')
for line in lines:
    line = line.strip()
    if not line:
        pdf.ln(2)
        continue
    
    # Check if it's a header (short lines in ALL CAPS or starting with key words)
    if line.isupper() and len(line) < 50:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, line, ln=True)
        pdf.set_font('Arial', '', 10)
    elif line.startswith('Part ') or line.startswith('Task ') or line.startswith('Prompt '):
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, line, ln=True)
        pdf.set_font('Arial', '', 10)
    else:
        # Regular text - wrap if too long
        pdf.multi_cell(0, 5, line)

# Save PDF
pdf.output('E:/IS392/Module_13_Prompt_Engineering_Assignment.pdf')
print('PDF created successfully')
