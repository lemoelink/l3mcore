import re

file_path = "plugins/telemetry_dashboard.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace CSS
old_css = """        :root {
            --bg-main: #020617;
            --bg-secondary: #0f172a;
            --bg-tertiary: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #3b82f6;
            --border-color: #334155;"""

new_css = """        :root {
            --bg-main: #0a0a0f;
            --bg-secondary: #14141c;
            --bg-tertiary: #1f1f2e;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #7c3aed;
            --border-color: #2a2a3e;"""

content = content.replace(old_css, new_css)

# Replace Title
content = content.replace("<title>l3mcore Business · Enterprise Telemetry</title>", "<title>l3mcore · Telemetry</title>")

# Replace Header Logo
old_logo = '<h1><span style="font-weight:300;">business.</span>LEMoE</h1>'
new_logo = '<h1>LEMoE<span style="color: var(--accent-color);">.</span></h1>'
content = content.replace(old_logo, new_logo)

# Replace Header Subtitle
old_sub = '<p style="color: var(--text-secondary); margin-top: 0.5rem;"><i class="fa-solid fa-server" style="color: var(--success);"></i> Live Enterprise Telemetry</p>'
new_sub = '<p style="color: var(--text-secondary); margin-top: 0.5rem;"><i class="fa-solid fa-server" style="color: var(--success);"></i> Live Telemetry</p>'
content = content.replace(old_sub, new_sub)

# Fix chart colors (from blue to purple)
content = content.replace("backgroundColor: '#3b82f6',", "backgroundColor: '#7c3aed',")
content = content.replace("backgroundColor: ['#3b82f6', '#a855f7', '#10b981', '#f59e0b', '#ef4444']", "backgroundColor: ['#7c3aed', '#60a5fa', '#10b981', '#f59e0b', '#ef4444']")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Telemetry dashboard updated successfully.")
