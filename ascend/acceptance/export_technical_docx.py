"""Export the project's plain Markdown headings, tables and code blocks to DOCX.

Standard-library OOXML export. Markdown remains the canonical review format.
"""
import argparse
import datetime
import re
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ET.register_namespace('w',W)
def node(name,attrs=None,text=None):
    e=ET.Element('{'+W+'}'+name,{'{'+W+'}'+k:v for k,v in (attrs or {}).items()})
    if text is not None:e.text=text
    return e

def plain(s):
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1（\2）',s)
    return s.replace('**','').replace('`','')

def paragraph(text,style='Normal',table=False):
    p=node('p');props=node('pPr');props.append(node('pStyle',{'val':style}));p.append(props)
    run=node('r')
    if table:
        rp=node('rPr');rp.append(node('sz',{'val':'18'}));run.append(rp)
    t=node('t',text=plain(text));t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    run.append(t);p.append(run);return p


def export(source,destination):
    doc=node('document');body=node('body');doc.append(body)
    lines=source.read_text().splitlines();index=0;code=False
    while index<len(lines):
        line=lines[index];index+=1
        if line.startswith('```'):code=not code;continue
        if code:body.append(paragraph(line,'Code'));continue
        if not line.strip() or line.strip()=='---':continue
        if line.startswith('|'):
            rows=[line]
            while index<len(lines) and lines[index].startswith('|'):rows.append(lines[index]);index+=1
            rows=[r for r in rows if not re.fullmatch(r'[|:\-\s]+',r)]
            table=node('tbl');props=node('tblPr');props.append(node('tblW',{'w':'0','type':'auto'}))
            borders=node('tblBorders')
            for edge in ['top','left','bottom','right','insideH','insideV']:borders.append(node(edge,{'val':'single','sz':'4','color':'B8C2CC'}))
            props.append(borders);table.append(props)
            for ri,row in enumerate(rows):
                tr=node('tr');trp=node('trPr');trp.append(node('cantSplit'))
                if ri==0:trp.append(node('tblHeader'))
                tr.append(trp)
                for value in row.strip('|').split('|'):
                    tc=node('tc')
                    if ri==0:
                        tcp=node('tcPr');tcp.append(node('shd',{'fill':'E8EEF7'}));tc.append(tcp)
                    tc.append(paragraph(value.strip(),table=True));tr.append(tc)
                table.append(tr)
            body.append(table);body.append(paragraph(''));continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));style='Title' if level==1 else f'Heading{min(level-1,3)}'
            body.append(paragraph(line.lstrip('#').strip(),style));continue
        if not re.match(r'^\s*(?:[-*]|\d+\.)\s',line):
            while index<len(lines) and lines[index].strip() and not re.match(r'^(?:[|#]|```|\s*[-*]\s|\d+\.\s)',lines[index]):
                line+=' '+lines[index];index+=1
        body.append(paragraph(line))
    section=node('sectPr');section.append(node('pgSz',{'w':'11906','h':'16838'}))
    section.append(node('pgMar',{'top':'1134','right':'1134','bottom':'1134','left':'1134','header':'567','footer':'567','gutter':'0'}));body.append(section)
    styles=node('styles')
    for name,size,bold in [('Normal',21,False),('Title',32,True),('Heading1',28,True),('Heading2',24,True),('Heading3',22,True),('Code',18,False)]:
        st=node('style',{'type':'paragraph','styleId':name});st.append(node('name',{'val':name}))
        if name!='Normal':st.append(node('basedOn',{'val':'Normal'}))
        pp=node('pPr');pp.append(node('spacing',{'after':'100','line':'300','lineRule':'auto'}))
        if name.startswith('Heading') or name=='Title':pp.append(node('keepNext'))
        st.append(pp);rp=node('rPr');rp.append(node('rFonts',{'ascii':'Consolas' if name=='Code' else 'Calibri','hAnsi':'Calibri','eastAsia':'黑体' if bold else '宋体'}));rp.append(node('sz',{'val':str(size)}))
        if bold:rp.append(node('b'))
        st.append(rp);styles.append(st)
    content_types='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
    root_rels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    doc_rels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml',content_types);archive.writestr('_rels/.rels',root_rels)
        archive.writestr('word/document.xml',ET.tostring(doc,encoding='utf-8',xml_declaration=True))
        archive.writestr('word/styles.xml',ET.tostring(styles,encoding='utf-8',xml_declaration=True))
        archive.writestr('word/_rels/document.xml.rels',doc_rels)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():ET.fromstring(archive.read(name))
        document=ET.fromstring(archive.read('word/document.xml'))
        text=''.join(t.text or '' for t in document.iter('{'+W+'}t'))
        assert len(text)>1000
    print(f'{destination}: OOXML structure valid; {len(text)} text characters')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();export(a.source,a.destination)

if __name__=='__main__':main()
