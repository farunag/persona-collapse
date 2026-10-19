import json,glob,os,re,collections,statistics as st
import pathlib
B=str(pathlib.Path(__file__).resolve().parent.parent/'data'/'prior')
def load(f): return json.load(open(f,encoding='utf-8'))
def key(f):
    m=re.search(r'(\w+)_p(\d+)_t(\d+)_(\d+_\d+)',os.path.basename(f)); return (m.group(1),int(m.group(2)),int(m.group(3)))
report=[]
for dom in ['coding','metacognitive','therapy']:
    en={key(f):f for f in glob.glob(f'{B}/raw/scaled-n60/{dom}/*.json')}
    langs={l:{key(f):f for f in glob.glob(f'{B}/{dom}_translated/{l}/*.json')} for l in ['ha','yo']}
    print(f'\n=== {dom}: en={len(en)} ha={len(langs["ha"])} yo={len(langs["yo"])}')
    print('  keys match en/ha/yo:', set(en)==set(langs['ha'])==set(langs['yo']))
    for l in ['en','ha','yo']:
        files=en if l=='en' else langs[l]
        nuser=[];moj=0;bad=[];chars=[];models=collections.Counter();aud=collections.Counter();empty=0;same_as_en=0;asst_en=0
        for k,f in files.items():
            d=load(f); c=d['conversation']
            models[d.get('model')]+=1; aud[d.get('auditor_model')]+=1
            u=[m['content'] for m in c if m['role']=='user']
            nuser.append(len(u)); chars+= [len(x) for x in u]
            txt=' '.join(u)
            if re.search('Æ|É|Ã|â€',txt): moj+=1
            empty+=sum(1 for x in u if not x.strip())
            if l!='en':
                e=[m['content'] for m in load(en[k])['conversation'] if m['role']=='user']
                if len(e)!=len(u): bad.append((os.path.basename(f),len(e),len(u)))
                same_as_en+=sum(1 for a,b in zip(u,e) if a.strip()==b.strip())
        print(f'  {l}: files={len(files)} user_turns min/med/max={min(nuser)}/{st.median(nuser)}/{max(nuser)} mojibake_files={moj} empty_turns={empty} untranslated(identical to en)={same_as_en} turncount_mismatch={len(bad)} mean_chars/turn={st.mean(chars):.0f}')
        print('     models',dict(models),'auditors',dict(aud))
        if bad[:3]: print('     mismatch examples',bad[:3])
