"""Admit only the known exact Kaggle save identity. No URL parsing or provider."""
import copy,hashlib,json,re

def need(ok,code):
    if not ok:raise ValueError(code)
def admit(response,owner,slug,*,kernel_id,version=1):
    need(type(response)is dict,'SAVE_RESPONSE_TYPE')
    need(type(owner)is str and re.fullmatch('[a-zA-Z0-9_-]+',owner)and type(slug)is str and re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug),'SAVE_EXPECTED_REF')
    need(type(kernel_id)is int and kernel_id>0 and type(version)is int and version==1,'SAVE_EXPECTED_VERSION_ID')
    ref=owner+'/'+slug;raw=response.get('ref');url='https://www.kaggle.com/code/'+ref
    need(raw in (ref,'/code/'+ref),'SAVE_EXACT_REF')
    need(response.get('url')==url,'SAVE_EXACT_URL')
    need(type(response.get('kernel_id'))is int and response['kernel_id']==kernel_id and type(response.get('version_number'))is int and response['version_number']==version,'SAVE_EXACT_VERSION_ID')
    need(not response.get('error') and not any(response.get(k)for k in ('invalid_tags','invalid_dataset_sources','invalid_kernel_sources','invalid_competition_sources','invalid_model_sources')),'SAVE_ERROR')
    preserved=copy.deepcopy(response)
    return {'identity':{'owner':owner,'slug':slug,'kernel_id':kernel_id,'version':version},'normalized_ref':ref,'raw_response':preserved,'normalization':'EXACT_CODE_PATH'if raw=='/code/'+ref else'ALREADY_CANONICAL','raw_response_sha256':hashlib.sha256(json.dumps(preserved,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
