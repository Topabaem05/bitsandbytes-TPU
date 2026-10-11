"""Official CLI display closure rules, copied from reviewed M5 reader-v4."""
import re
import contract as C
def output_lines(text):
 C.require(isinstance(text,str),'CLI_RAW_TEXT')
 clean=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
 return [line.strip()for line in clean.splitlines()if line.strip()]


def sessions_output(text):
 lines=output_lines(text);pruned=0
 if lines and re.fullmatch(r'\[colab\] Pruned [1-9][0-9]* stale local session\(s\)\.',lines[0]):
  pruned=int(lines.pop(0).split()[2])
 empty=lines==['[colab] No active sessions found on server.']
 rows=[]
 if not empty:
  C.require(bool(lines),'SESSIONS_RAW_REQUIRED')
  for line in lines:
   match=re.fullmatch(r'\[([A-Za-z0-9][A-Za-z0-9_-]{0,127}|\?)\] ([A-Za-z0-9_-]{1,256}) \| Hardware: ([A-Za-z0-9_-]+) \| Shape: ([A-Za-z0-9 _-]+) \| Variant: ([A-Za-z0-9_-]+)',line)
   C.require(match is not None,'OFFICIAL_SESSIONS_RAW')
   rows.append(dict(zip(('session','endpoint','hardware','shape','variant'),match.groups())))
 C.require(len({(r['session'],r['endpoint'])for r in rows})==len(rows),'DUPLICATE_SESSION_ROW')
 return {'empty':empty,'sessions':rows,'pruned_local_sessions':pruned}


def usage_output(text):
 lines=output_lines(text)
 patterns=(r'Current balance: (-?[0-9]+\.[0-9]{2}) compute units',r'Usage rate: ([0-9]+\.[0-9]{2})/hr',r'Active assignments: ([0-9]+)')
 C.require(len(lines)==3,'OFFICIAL_USAGE_LABELS')
 values=[]
 for line,pattern in zip(lines,patterns):
  match=re.fullmatch(pattern,line);C.require(match is not None,'OFFICIAL_USAGE_LABELS');values.append(match[1])
 return {'displayed_balance':values[0],'displayed_rate_per_hour':values[1],'active_assignments':int(values[2]),'displayed_zero':values[1]=='0.00'and int(values[2])==0}


def resource_empty(session_text,usage_text):
 sessions=sessions_output(session_text);usage=usage_output(usage_text)
 C.require(sessions['empty']and usage['displayed_zero'],'PHYSICAL_EMPTY_SERVER_ZERO_DISPLAYED_USAGE')
 return {'sessions':sessions,'usage':usage}


