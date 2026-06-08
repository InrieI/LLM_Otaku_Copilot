import json
from pathlib import Path
from collections import defaultdict

project = r'E:\Inriel_Project\LLM_OtakuWifu_Copilot'
this_session = r'C:\Users\26974\.claude\projects\E--Inriel-Project-LLM-OtakuWifu-Copilot\8a5360ae-87ea-4ed6-9e9f-9047c53cf919.jsonl'

# Step 1: Get the Write at line 421 (live2d.html base state = start of continued session)
live2d_base = None
with open(this_session, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 421:
            obj = json.loads(line)
            msg = obj.get('message', {})
            content = msg.get('content', [])
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_use':
                    inp = block.get('input', {})
                    if block.get('name') == 'Write':
                        live2d_base = inp.get('content', '')
            break

print(f'live2d.html base from Write at line 421: {len(live2d_base)} chars')

# Step 2: Collect edits from previous sessions (all) and this session (before line 291 only)
# Exclude live2d.html edits - we use the Write at 421 directly
sessions = [
    r'C:\Users\26974\.claude\projects\E--Inriel-Project-LLM-OtakuWifu-Copilot\fa63547e-3a5c-4423-b6f7-35683385b976.jsonl',
    r'C:\Users\26974\.claude\projects\E--Inriel-Project-LLM-OtakuWifu-Copilot\2ec32d89-3080-4965-93e6-482d0ed8d612.jsonl',
    r'C:\Users\26974\.claude\projects\E--Inriel-Project-LLM-OtakuWifu-Copilot\16a3a0f9-8605-4647-b2e6-c0e15a911658.jsonl',
]

all_edits = []

for spath in sessions:
    with open(spath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                msg = obj.get('message', {})
                content = msg.get('content', [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'tool_use':
                            name = block.get('name', '')
                            inp = block.get('input', {})
                            if name in ('Write', 'Edit') and 'file_path' in inp:
                                fp = inp['file_path']
                                if fp.startswith(project):
                                    rel = fp[len(project)+1:]
                                    if 'live2d' in rel:
                                        continue
                                    if name == 'Write':
                                        all_edits.append({'op': 'Write', 'file': rel, 'content': inp.get('content', '')})
                                    else:
                                        all_edits.append({'op': 'Edit', 'file': rel, 'old': inp.get('old_string', ''), 'new': inp.get('new_string', ''), 'replace_all': inp.get('replace_all', False)})
            except:
                pass

# This session - edits before line 291, exclude live2d.html
with open(this_session, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 291:
            break
        try:
            obj = json.loads(line)
            msg = obj.get('message', {})
            content = msg.get('content', [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        name = block.get('name', '')
                        inp = block.get('input', {})
                        if name in ('Write', 'Edit') and 'file_path' in inp:
                            fp = inp['file_path']
                            if fp.startswith(project):
                                rel = fp[len(project)+1:]
                                if 'live2d' in rel:
                                    continue
                                if name == 'Write':
                                    all_edits.append({'op': 'Write', 'file': rel, 'content': inp.get('content', '')})
                                else:
                                    all_edits.append({'op': 'Edit', 'file': rel, 'old': inp.get('old_string', ''), 'new': inp.get('new_string', ''), 'replace_all': inp.get('replace_all', False)})
        except:
            pass

print(f'Total edits for other files: {len(all_edits)}')

# Apply edits
by_file = defaultdict(list)
for e in all_edits:
    by_file[e['file']].append(e)

for fname in sorted(by_file.keys()):
    fpath = Path(project) / fname
    if fpath.exists():
        current = fpath.read_text(encoding='utf-8')
    else:
        current = ''
    ops = by_file[fname]
    success = fail = 0
    for e in ops:
        if e['op'] == 'Write':
            current = e['content']
            success += 1
        elif e['op'] == 'Edit':
            if e['old'] in current:
                if e.get('replace_all'):
                    current = current.replace(e['old'], e['new'])
                else:
                    current = current.replace(e['old'], e['new'], 1)
                success += 1
            else:
                fail += 1
    print(f'{fname}: {success} ok, {fail} failed / {len(ops)} total')
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(current, encoding='utf-8')

# Write live2d.html from the Write at line 421
(Path(project) / 'frontend' / 'live2d.html').write_text(live2d_base, encoding='utf-8')
print(f'frontend/live2d.html: restored from Write at line 421 ({len(live2d_base)} chars)')

print()
print('Done.')
