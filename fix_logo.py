import re

files = [
    "web/business/index.html",
    "web/business/index_en.html"
]

# The logo pattern for navbar and footer
# <div class="logo"><a ...>LEMoE <span style="color: var(--accent-color); font-weight: 300;">Business</span></a></div>
# <div class="logo">LEMoE <span style="color: var(--accent-color); font-weight: 300;">Business</span></div>

old_nav_es = '<div class="logo"><a href="index.html" style="text-decoration: none; color: inherit;">LEMoE <span style="color: var(--accent-color); font-weight: 300;">Business</span></a></div>'
new_nav_es = '<div class="logo"><a href="index.html" style="text-decoration: none; color: inherit;"><span style="color: var(--accent-color); font-weight: 300;">business.</span>LEMoE</a></div>'

old_nav_en = '<div class="logo"><a href="index_en.html" style="text-decoration: none; color: inherit;">LEMoE <span style="color: var(--accent-color); font-weight: 300;">Business</span></a></div>'
new_nav_en = '<div class="logo"><a href="index_en.html" style="text-decoration: none; color: inherit;"><span style="color: var(--accent-color); font-weight: 300;">business.</span>LEMoE</a></div>'

old_footer = '<div class="logo">LEMoE <span style="color: var(--accent-color); font-weight: 300;">Business</span></div>'
new_footer = '<div class="logo"><span style="color: var(--accent-color); font-weight: 300;">business.</span>LEMoE</div>'

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    content = content.replace(old_nav_es, new_nav_es)
    content = content.replace(old_nav_en, new_nav_en)
    content = content.replace(old_footer, new_footer)
    
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)

print("Logo updated successfully.")
