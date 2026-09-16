#!/usr/bin/env python3
# fetch_source.py — 抓取外部教材页面正文（用于事实核验，非二手转述）
# 用法: python3 _meta/tools/fetch_source.py <URL> [输出字符数]
# 结果同时存为 ./_<文件名>.txt 便于核对
import re,html,sys,urllib.request,pathlib
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'})
    try:
        raw=urllib.request.urlopen(req,timeout=30).read().decode('utf-8','ignore')
    except Exception as e:
        try:
            raw=urllib.request.urlopen(urllib.request.Request("https://r.jina.ai/"+url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40).read().decode('utf-8','ignore')
        except Exception as e2:
            return None,"FETCH_FAIL %s / %s"%(e,e2)
    t=re.sub(r'<script.*?</script>|<style.*?</style>','',raw,flags=re.S)
    t=re.sub(r'<[^>]+>',' ',t)
    t=html.unescape(t); t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t)
    return t,None
if __name__=='__main__':
    url=sys.argv[1]; limit=int(sys.argv[2]) if len(sys.argv)>2 else 4000
    t,err=get(url)
    if err: print("ERR",url,err); sys.exit(1)
    pathlib.Path('_'+url.rstrip('/').split('/')[-1]+'.txt').write_text(t)
    print(t[:limit])
