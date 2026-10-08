"""Independent protobuf wire encoder for synthetic mutation controls."""
def number(raw,pos):
    value=0;shift=0
    while True:
        item=raw[pos];pos+=1;value|=(item&127)<<shift
        if not item&128:return value,pos
        shift+=7

def encode_number(value):
    result=bytearray()
    while value>127:result.append((value&127)|128);value>>=7
    result.append(value);return bytes(result)

def decode(raw):
    pos=0;result=[]
    while pos<len(raw):
        tag,pos=number(raw,pos);kind=tag&7
        if kind==0:value,pos=number(raw,pos)
        elif kind==2:size,pos=number(raw,pos);value=raw[pos:pos+size];pos+=size
        elif kind in (1,5):size={1:8,5:4}[kind];value=raw[pos:pos+size];pos+=size
        else:raise AssertionError(kind)
        result.append([tag>>3,kind,value])
    assert pos==len(raw);return result

def encode(items):
    out=bytearray()
    for field,kind,value in items:
        out+=encode_number((field<<3)|kind)
        if kind==0:out+=encode_number(value)
        elif kind==2:out+=encode_number(len(value))+value
        else:out+=value
    return bytes(out)

def edit(raw,path,operation):
    items=decode(raw)
    if not path:return encode(operation(items))
    field,ordinal=path[0];indexes=[i for i,row in enumerate(items) if row[0]==field];index=indexes[ordinal]
    assert items[index][1]==2;items[index][2]=edit(items[index][2],path[1:],operation);return encode(items)

def replace(items,field,kind,value):
    indexes=[i for i,row in enumerate(items) if row[0]==field]
    assert len(indexes)<=1
    row=[field,kind,value]
    if indexes:items[indexes[0]]=row
    else:items.append(row)
    return items

