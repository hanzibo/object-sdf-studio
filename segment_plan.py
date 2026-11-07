"""Persistent segmentation boundaries and independently reusable segment results (stdlib only)."""
from pathlib import Path
import hashlib,json,os

PLAN_FILE='metadata/segmentation_plan.json'
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temp,path)
def read_plan(root):
    p=Path(root)/PLAN_FILE
    return json.loads(p.read_text()) if p.exists() else dict(enabled=False,interval=100,starts=[0])
def ranges(files,plan):
    starts=sorted(set(plan.get('starts',[0])))
    if not starts or starts[0]!=0 or any(type(i)!=int or not 0<=i<len(files) for i in starts):
        raise ValueError('分段断点无效，请重新生成断点')
    return [(s,starts[i+1] if i+1<len(starts) else len(files)) for i,s in enumerate(starts)]
def signature(files,prompts,start,end):
    return digest(dict(files=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files[start:end]],
                       prompts={p.name:prompts[p.name] for p in files[start:end] if prompts.get(p.name)}))
def plan_signature(files,prompts,plan):
    return digest(dict(ranges=ranges(files,plan),segments=[signature(files,prompts,s,e) for s,e in ranges(files,plan)]))
def cache_file(root,files,start):
    return Path(root)/'metadata/segment_cache'/(digest(files[start].name)+'.json')
def cached_result(root,files,prompts,start,end):
    path=cache_file(root,files,start)
    try:
        result=json.loads(path.read_text());folder=Path(root)/result['masks']
        if result['signature']!=signature(files,prompts,start,end):return None
        if not all((folder/p.name).is_file() for p in files[start:end]):return None
        return result
    except (OSError,KeyError,ValueError):return None
