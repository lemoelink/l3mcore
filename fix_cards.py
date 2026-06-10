import re

files = [
    "web/business/index.html",
    "web/business/index_en.html"
]

old_card = 'style="background: var(--bg-main); padding: 2rem; border-radius: 12px; border: 1px solid var(--border-color);"'
new_card = 'class="premium-card"'

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    content = content.replace(old_card, new_card)
    
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)

print("Cards updated successfully.")
